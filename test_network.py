'''
This code runs pre-trained MGN.

If you use this code please cite:
"Multi-Garment Net: Learning to Dress 3D People from Images", ICCV 2019

Code author: Bharat
'''
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pickle as pkl

from network.base_network import PoseShapeOffsetModel
from config_ver1 import config, NUM, IMG_SIZE, FACE

TEMPLATE = pkl.load(open('assets/allTemplate_withBoundaries_symm.pkl', 'rb'),
                    encoding="latin1")

PCA_VERTS = {}
for garment in config.garmentKeys:
    with open(os.path.join('assets/garment_basis_35_temp20',
                           garment + '_param_{}_corrected.pkl'.format(config.PCA_)),
              'rb') as f:
        PCA_VERTS[garment] = pkl.load(f, encoding="latin1")


def pca2offsets(pca_layers, scatter_layers, pca_coeffs, naked_verts, vertexlabel,
                return_all=False):
    disps = []
    for l, s, p in zip(pca_layers, scatter_layers, pca_coeffs):
        temp = l(p)
        temp = s(temp)
        disps.append(temp)
    temp = tf.stack(disps, axis=-1)
    temp = tf.concat([tf.keras.backend.expand_dims(naked_verts, -1), temp], axis=-1)
    temp2 = tf.transpose(temp, perm=[0, 1, 3, 2])
    temp = tf.batch_gather(temp2, tf.cast(vertexlabel, tf.int32))
    temp = tf.squeeze(tf.transpose(temp, perm=[0, 1, 3, 2]))
    if return_all:
        return temp, temp2
    return temp


def split_garments(pca, mesh, vertex_label, gar):
    '''
    Since garments are layered we do net get high frequency parts for invisible garment vertices.
    Hence we generate the base garment from pca predictions and add the hf term whenever available.
    :param pred_mesh:
    :param garments:
    :return:
    '''
    vertex_label = vertex_label.reshape(-1, )
    base = PCA_VERTS[config.garmentKeys[gar]].inverse_transform(pca).reshape(-1, 3)
    ind = np.where(TEMPLATE[config.garmentKeys[gar]][1])[0]
    gar_mesh = Mesh(mesh.v, mesh.f)

    gar_mesh.v[ind] = base
    gar_mesh.v[vertex_label] = mesh.v[vertex_label]
    gar_mesh.keep_vertices(ind)
    return gar_mesh


def get_results(m, inp, with_pose=False):
    images = [inp[f'image_{i}'].astype('float32') for i in range(NUM)]
    J_2d = [inp[f'J_2d_{i}'].astype('float32') for i in range(NUM)]
    vertex_label = inp['vertexlabel'].astype('int64')

    out = m([images, vertex_label, J_2d])

    with open('assets/hresMapping.pkl', 'rb') as f:
        _, faces = pkl.load(f, encoding="latin1")

    pca_layers = [l.PCA_ for l in m.garmentModels]
    scatter_layers = m.scatters
    pca_coeffs = np.transpose(out['pca_verts'], [1, 0, 2])
    naked_verts = out['vertices_naked']
    temp = pca2offsets(pca_layers, scatter_layers, pca_coeffs,
                       naked_verts.numpy().astype('float32'), vertex_label)

    pred_mesh = Mesh(out['vertices_tposed'][0].numpy(), faces)
    pred_naked = Mesh(naked_verts[0].numpy(), faces)
    pred_pca = Mesh(temp[0].numpy(), faces)

    gar_meshes = []
    for gar in np.unique(inp['vertexlabel'][0]):  # np.where(inp['garments'][0])[0]:
        if gar == 0:
            continue
        gar_meshes.append(split_garments(out['pca_verts'][0][gar - 1], pred_mesh,
                                         vertex_label[0] == gar, gar - 1))

    return {'garment_meshes': gar_meshes, 'body': pred_naked, 'pca_mesh': pred_pca}


def load_model(model_dir):
    m = PoseShapeOffsetModel(config,
                             latent_code_garms_sz=int(config.latent_code_garms_sz / 2))

    # Create the models and optimizers.
    model_objects = {
        'network': m,
        'optimizer': m.optimizer,
        'step': tf.Variable(0),
    }
    latest_cpkt = tf.train.latest_checkpoint(model_dir)
    if latest_cpkt:
        print(('Using latest checkpoint at ' + latest_cpkt))
    else:
        print('No pre-trained model found')
    checkpoint = tf.train.Checkpoint(**model_objects)

    # Restore variables on creation if a checkpoint exists.
    checkpoint.restore(latest_cpkt)

    return m


def fine_tune(m, inp, out, display=False):
    ## Need to do a forward pass to get trainable variables
    images = [inp[f'image_{i}'].astype('float32') for i in range(NUM)]
    vertex_label = inp['vertexlabel'].astype('int64')
    J_2d = [inp[f'J_2d_{i}'].astype('float32') for i in range(NUM)]

    _ = m([images, vertex_label, J_2d])

    ## First optimize pose then other stuff
    vars = []
    losses_2d = {}
    epochs = config.train.epochs
    vars = ['pose_trans']
    losses_2d['rendered'] = 5 * 10. ** 3
    losses_2d['laplacian'] = 5 * 10 ** 5.
    for i in range(NUM):
        losses_2d['J_2d_{}'.format(i)] = 10 ** 3.
    vars2opt = []
    for v in vars:
        for vv in m.trainable_variables:
            if v in vv.name:
                vars2opt.append(vv.name)

    for ep in range(epochs):
        lo = m.train(inp, out, loss_dict=losses_2d, vars2opt=vars2opt)
        J_2d = 0
        stri = ''
        for k in losses_2d:
            if 'J_2d' in k:
                J_2d += abs(lo[k])
                continue
            stri = stri + k + ' :{}, '.format(lo[k])
        stri = stri + 'J_2d' + ' :{}'.format(J_2d / NUM)
        print(('Ep: {}, {}'.format(ep, stri)))

    vars.extend(['pca_comp', 'betas', 'latent_code_offset_ShapeMerged', 'byPass'])
    losses_2d['laplacian'] = 5 * 10 ** 5.
    losses_2d['rendered'] = 5 * 10. ** 5
    for i in range(NUM):
        losses_2d['J_2d_{}'.format(i)] = 10.

    vars2opt = []
    for v in vars:
        for vv in m.trainable_variables:
            if v in vv.name:
                vars2opt.append(vv.name)

    for ep in range(epochs):
        lo = m.train(inp, out, loss_dict=losses_2d, vars2opt=vars2opt)
        J_2d = 0
        stri = ''
        for k in losses_2d:
            if 'J_2d' in k:
                J_2d += abs(lo[k])
                continue
            stri = stri + k + ' :{}, '.format(lo[k])
        stri = stri + 'J_2d' + ' :{}'.format(J_2d / NUM)
        print(('Ep: {}, {}'.format(ep, stri)))

    return m

if __name__ == "__main__":
    import os
    from os.path import exists, join, split
    from psbody.mesh import Mesh, MeshViewer, MeshViewers

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    tf.enable_eager_execution()

    model_dir = 'saved_model/'
    ## Load model
    m = load_model(model_dir)

    ## Load test data
    print("Load test data")
    dat = pkl.load(open('assets/test_data.pkl', "rb"), encoding="latin1")
    dat = {k: v[:config.train.batch_size] for k, v in dat.items()}

    ## Get results before optimization
    print("Get results before optimization")
    pred = get_results(m, dat)

    import code

    code.interact(local=locals())
    mv = MeshViewers((1, 2), keepalive=True)
    mv[0][0].set_static_meshes(pred['garment_meshes'] + [pred['body']])
    mv[0][1].set_static_meshes([pred['body']])

    ## Optimize the network
    print("Optimize the network")
    m = fine_tune(m, dat, dat, display=False)
    pred = get_results(m, dat, )

    mv1 = MeshViewers((1, 2), keepalive=True)
    mv1[0][0].set_static_meshes(pred['garment_meshes'])
    mv1[0][1].set_static_meshes([pred['body']])

    print('Done')
