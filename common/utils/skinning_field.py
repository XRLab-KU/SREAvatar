import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.ops import knn_points
from utils.smflix_lib.smflix.lbs import lbs

# Builds the diffused skinning field in memory at startup (~0.7 s) instead of shipping it as a 169 MB file.
# Same algorithm as PERSONA's tools/diffused_skinning_weights/make_skinning_weight.py.
# The SMFLIX arrays used here (weights, v_template, J_regressor, kintree_table, posedirs) are identical to SMPLX_NEUTRAL.npz.

def create_diffused_skinning_field(verts, smplx_weights, grid_res=128, smooth_iters=100, proximity_thresh=0.02, expansion_factor=0.3):
    joint_num = smplx_weights.shape[1]
    finger_joint_idx = list(range(25, 55)) # SMPL-X finger joints
    non_finger_idx = [i for i in range(joint_num) if i not in finger_joint_idx]

    # extended bounding box
    vmin, vmax = verts.min(0).values, verts.max(0).values
    margin = expansion_factor * (vmax - vmin)
    vmin = vmin - margin
    vmax = vmax + margin

    # create 3D voxel grid
    coords = [torch.linspace(vmin[i], vmax[i], grid_res).cuda() for i in range(3)]
    xx, yy, zz = torch.meshgrid(*coords, indexing='ij')
    grid_points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3) # [D^3, 3]

    # find points close to surface
    dists = knn_points(grid_points[None], verts[None], K=1).dists[0, :, 0]
    surface_mask = dists < (proximity_thresh ** 2)
    surface_points = grid_points[surface_mask]

    # get weights from nn
    nn_idx = knn_points(surface_points[None], verts[None], K=1).idx[0, :, 0]
    surface_weights = smplx_weights[nn_idx] # [Ns, J]

    # initialize weight field
    field = torch.zeros((joint_num, grid_res, grid_res, grid_res)).cuda()
    mask = surface_mask.reshape(grid_res, grid_res, grid_res)
    field[:, mask] = surface_weights.T

    # diffuse every joint except fingers
    kernel = torch.tensor([[[0,0,0],[0,1,0],[0,0,0]],
                           [[0,1,0],[1,-6,1],[0,1,0]],
                           [[0,0,0],[0,1,0],[0,0,0]]],
                          dtype=torch.float32).cuda()[None, None].repeat(len(non_finger_idx), 1, 1, 1, 1)

    field_diffuse = field[non_finger_idx].clone()
    fixed_val = torch.zeros_like(field_diffuse)
    fixed_val[:, mask] = surface_weights[:, non_finger_idx].T
    fixed_mask = mask[None].repeat(len(non_finger_idx), 1, 1, 1)

    for _ in range(smooth_iters): # Laplacian smoothing
        field_diffuse_pad = F.pad(field_diffuse, (1,1,1,1,1,1), mode='replicate')
        lap = F.conv3d(field_diffuse_pad[None], kernel, groups=len(non_finger_idx))[0]
        field_diffuse = field_diffuse + 0.1 * lap
        field_diffuse = torch.where(fixed_mask, fixed_val, field_diffuse)

    field[non_finger_idx] = field_diffuse

    # normalize
    field = torch.clamp(field, min=0)
    field = field / (field.sum(0, keepdim=True) + 1e-8)
    field = field.permute(1, 2, 3, 0).contiguous() # [D, D, D, J]
    return field, coords

def make_skinning_field(model_npz_path):
    # neutral shape in the 大 pose, posed the way smplx.create()'s default layer does it in PERSONA
    data = np.load(model_npz_path, allow_pickle=True)
    v_template = torch.FloatTensor(data['v_template']).cuda()
    weights = torch.FloatTensor(data['weights']).cuda()
    J_regressor = torch.FloatTensor(data['J_regressor']).cuda()
    posedirs = torch.FloatTensor(data['posedirs'].reshape(-1, data['posedirs'].shape[-1]).T).cuda() # (P, V*3)
    parents = torch.LongTensor(data['kintree_table'][0].astype(np.int64)).cuda()
    parents[0] = -1

    body_pose = torch.zeros((21,3))
    body_pose[0] = torch.FloatTensor([0, 0, 1/3])
    body_pose[1] = torch.FloatTensor([0, 0, -1/3])
    # root, body, jaw, eyes are zero; hands use the mean pose (smplx default: use_pca with zero coeffs, flat_hand_mean=False)
    full_pose = torch.cat([torch.zeros((1,3)), body_pose, torch.zeros((3,3)),
                           torch.FloatTensor(data['hands_meanl']).view(-1,3), torch.FloatTensor(data['hands_meanr']).view(-1,3)]).view(1,-1).cuda()

    betas = torch.zeros((1,1)).cuda()
    shapedirs = torch.zeros((v_template.shape[0],3,1)).cuda()
    verts, _ = lbs(betas, full_pose, v_template[None], shapedirs, posedirs, J_regressor, None, parents, weights)

    with torch.no_grad():
        field, coords = create_diffused_skinning_field(verts[0].detach(), weights)
    return field, coords