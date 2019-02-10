import numpy as np
import Cn.utils as utils
import scipy
import pickle
import os
import aspire.abinitio as abinitio
from tqdm import tqdm
from Cn.config_symm import AbinitioSymmConfig


def estimate_relative_viewing_directions(npf, cache_file_name=None, rots_gt=None):
    if AbinitioSymmConfig.n_symm in [3, 4]:
        print('Estimating relative viewing directions for n<=4')
        viis, vijs = estimate_relative_viewing_directions_c3_c4(npf, rots_gt)
    else:
        print('Estimating relative viewing directions for n>4')
        viis, vijs = estimate_relative_viewing_directions_cn(npf, cache_file_name, rots_gt)
    if rots_gt is not None:
        utils.detection_rate_viis(viis, AbinitioSymmConfig.n_symm, rots_gt)
        utils.detection_rate_vijs(vijs, AbinitioSymmConfig.n_symm, rots_gt)
    return viis, vijs


def estimate_relative_viewing_directions_c3_c4(npf, rots_gt=None):
    n_symm = AbinitioSymmConfig.n_symm
    n_theta = AbinitioSymmConfig.n_theta
    max_shift = AbinitioSymmConfig.max_shift
    shift_step = AbinitioSymmConfig.shift_step
    if AbinitioSymmConfig.is_use_gt:
        assert rots_gt is not None
        clmatrix = utils.find_cl_gt(n_symm, n_theta, rots_gt, is_simulate_J=True, single_cl=True)
        sclmatrix = utils.find_scl_gt(n_symm, n_theta, rots_gt, is_simulate_J=True, is_simulate_transpose=True)
    else:
        print('estimating common-lines')
        clmatrix, *_ = abinitio.cryo_clmatrix_cpu_pystyle(npf, max_shift, shift_step)
        print('estimating self common-lines')
        sclmatrix, *_ = find_scl(npf)

    print('estimating relative orientations')
    Rijs = abinitio.cryo_syncmatrix_vote_3n(clmatrix, n_theta)
    print('estimating self relative orientations')
    Riis = estimate_self_relative_rots(sclmatrix, rots_gt)
    if rots_gt is not None:
        utils.cl_detection_rate_single(n_symm, clmatrix, rots_gt, n_theta, AbinitioSymmConfig.angle_tol_err_deg)
        utils.scl_detection_rate(n_symm, sclmatrix, rots_gt, n_theta, AbinitioSymmConfig.angle_tol_err_deg)
        utils.detection_rate_self_relative_rots(Riis, n_symm, rots_gt)
        utils.detection_rate_relative_rots(Rijs, n_symm, rots_gt)
    print('local handedness')
    viis, vijs = local_handedness_sync(Riis, Rijs)
    return viis, vijs


def estimate_self_relative_rots(sclmatrix, rots_gt):
    n_symm = AbinitioSymmConfig.n_symm
    n_theta = AbinitioSymmConfig.n_theta
    assert n_symm == 3 or n_symm == 4, "supports only C3 or C4"

    cos_diff = np.cos((sclmatrix[:, 1] - sclmatrix[:, 0]) * 2 * np.pi / n_theta)

    if n_symm == 3:
        # cos_diff is supposed to be <= 0.5, but due to discretization that might be violated
        if np.max(cos_diff) > 0.5:
            print("cos(angular_diff) should be < 0.5. maximum found=%f. number of estimates exceeding=%d"
                  % (np.max(cos_diff), np.count_nonzero(cos_diff > 0.5)))
            cos_diff[cos_diff > 0.5] = 0.5
        gammas = np.arccos(cos_diff / (1 - cos_diff))
    else:
        # cos_diff is supposed to be <= 0, but due to discretization that might be violated
        if np.max(cos_diff) > 0:
            print("cos(angular-diff) should be < 0. maximum found=%f. number of estimates exceeding=%d"
                  % (np.max(cos_diff), np.count_nonzero(cos_diff > 0)))
            cos_diff[cos_diff > 0] = 0
        gammas = np.arccos((1 + cos_diff) / (1 - cos_diff))

    if rots_gt is not None:
        dr = utils.detection_rate_gammas(gammas, n_symm, rots_gt, angle_deg_tol_err=10)
        print("Rii detection rate gammas=%.2f, n_symm = %d" % (dr, n_symm))

    # calculate the remaining euler angles
    aa = sclmatrix[:, 0] * 2 * np.pi / n_theta
    bb = sclmatrix[:, 1] * 2 * np.pi / n_theta + np.pi  # TODO: get rid of the addition of pi by fixing ang_to_orth

    Riis = utils.ang_to_orth(-bb, gammas, aa)
    return Riis


def local_handedness_sync(Riis, Rijs):
    n_symm = AbinitioSymmConfig.n_symm
    assert n_symm == 3 or n_symm == 4

    n_images = len(Riis)
    assert scipy.special.comb(n_images, 2) == len(Rijs)

    viis = np.zeros((n_images, 3, 3))

    for i, Rii in enumerate(Riis):
        viis[i] = np.mean([np.linalg.matrix_power(Rii, s) for s in np.arange(n_symm)], axis=0)

    m_choose_2 = scipy.special.comb(n_images, 2).astype(int)
    vijs = np.zeros((m_choose_2, 3, 3))
    e1 = [1, 0, 0]
    opts = np.zeros((8, 3, 3))  # holds all inner sync options per i<j
    scores_rank1 = np.zeros(8)
    min_idxs = np.zeros((m_choose_2, 3, 3))
    c = 0
    for i in np.arange(n_images):
        for j in np.arange(i + 1, n_images):
            Rii = Riis[i]
            Rjj = Riis[j]
            Rij = Rijs[c]

            Rii_J = utils.J_conjugate(Rii)
            Rjj_J = utils.J_conjugate(Rjj)
            # testing 8 possibilities: 
            # a. whether or not to transpose Rii (so that Rii and Rjj match)
            # b. whether or not to J-conjugate Rii (so that it matches to Rij J-con class)
            # c. whether or not to J-conjugate Rjj (so that it matches to Rij J-con class)
            if n_symm == 3:
                opts[0] = Rij + np.linalg.multi_dot([Rii, Rij, Rjj]) + np.linalg.multi_dot([Rii.T, Rij, Rjj.T])
                opts[1] = Rij + np.linalg.multi_dot([Rii_J, Rij, Rjj]) + np.linalg.multi_dot([Rii_J.T, Rij, Rjj.T])
                opts[2] = Rij + np.linalg.multi_dot([Rii, Rij, Rjj_J]) + np.linalg.multi_dot([Rii.T, Rij, Rjj_J.T])
                opts[3] = Rij + np.linalg.multi_dot([Rii_J, Rij, Rjj_J]) + np.linalg.multi_dot([Rii_J.T, Rij, Rjj_J.T])

                opts[4] = Rij + np.linalg.multi_dot([Rii.T, Rij, Rjj]) + np.linalg.multi_dot([Rii, Rij, Rjj.T])
                opts[5] = Rij + np.linalg.multi_dot([Rii_J.T, Rij, Rjj]) + np.linalg.multi_dot([Rii_J, Rij, Rjj.T])
                opts[6] = Rij + np.linalg.multi_dot([Rii.T, Rij, Rjj_J]) + np.linalg.multi_dot([Rii, Rij, Rjj_J.T])
                opts[7] = Rij + np.linalg.multi_dot([Rii_J.T, Rij, Rjj_J]) + np.linalg.multi_dot([Rii_J, Rij, Rjj_J.T])

                opts = opts / 3  # normalize
            else:
                opts[0] = Rij + np.linalg.multi_dot([Rii, Rij, Rjj])
                opts[1] = Rij + np.linalg.multi_dot([Rii_J, Rij, Rjj])
                opts[2] = Rij + np.linalg.multi_dot([Rii, Rij, Rjj_J])
                opts[3] = Rij + np.linalg.multi_dot([Rii_J, Rij, Rjj_J])

                opts[4] = Rij + np.linalg.multi_dot([Rii.T, Rij, Rjj])
                opts[5] = Rij + np.linalg.multi_dot([Rii_J.T, Rij, Rjj])
                opts[6] = Rij + np.linalg.multi_dot([Rii.T, Rij, Rjj_J])
                opts[7] = Rij + np.linalg.multi_dot([Rii_J.T, Rij, Rjj_J])

                opts = opts / 2  # normalize

            for k, opt in enumerate(opts):
                _, svals, _ = np.linalg.svd(opt)
                scores_rank1[k] = np.linalg.norm(svals - e1, 2)
            min_idx = np.argmin(scores_rank1)
            vijs[c] = opts[min_idx]
            min_idxs[c] = min_idx
            c += 1

    hist, _ = np.histogram(min_idxs, np.arange(9))
    print("hist local handedness=" + str(hist))

    return viis, vijs


def find_scl(npf):
    # the angle between self-common-lines is [60, 180] (for C3) and [90,180] (for C4) but since antipodal
    # lines are perfectly correlated we mustn't test angles too close to 180 degrees apart
    n_symm = AbinitioSymmConfig.n_symm
    n_theta = AbinitioSymmConfig.n_theta
    n_r = AbinitioSymmConfig.n_r
    max_shift_1d = np.ceil(2 * np.sqrt(2) * AbinitioSymmConfig.max_shift)
    shift_step = AbinitioSymmConfig.shift_step
    if n_theta % 2 == 1:
        raise ValueError('n_theta must be even')
    if n_symm not in [3, 4]:
        raise ValueError('n_symm must be either 3 or 4')
    if n_symm == 3:
        min_angle_diff = 60*np.pi/180  # TODO: extract these angles as params
        max_angle_diff = 165*np.pi/180
    else:  # i.e., n_symm == 4
        min_angle_diff = 90*np.pi/180
        max_angle_diff = 160*np.pi/180

    n_images = len(npf)
    sclmatrix = np.zeros((n_images, 2))
    corrs_stats = np.zeros(n_images)
    shifts_stats = np.zeros(n_images)
    X, Y = np.meshgrid(range(n_theta), range(n_theta//2))
    diff = Y - X
    unsigned_angle_diff = np.arccos(np.cos(diff*2*np.pi/n_theta))

    good_diffs = np.logical_and(min_angle_diff < unsigned_angle_diff, unsigned_angle_diff < max_angle_diff)

    shift_phases = utils.calc_shift_phases(n_r, max_shift_1d, shift_step)
    n_shifts = len(shift_phases)
    for i in range(n_images):
        npf_i = npf[i]

        # generate all shifted versions of the images
        npf_i_half = npf_i[:n_theta // 2]

        npf_i_half_shifted = np.array([npf_i_half*shift_phase for shift_phase in shift_phases])  # shape is (n_shifts, n_theta/2, n_r)
        npf_i_half_shifted = np.reshape(npf_i_half_shifted, (-1, n_r))  # shape is (n_theta/2 * n_shifts, n_r)

        # ignoring dc-term.
        npf_i[:, 0] = 0
        npf_i_half_shifted[:, 0] = 0

        # normalize each ray to have norm equal to 1
        npf_i = np.array([ray/np.linalg.norm(ray) for ray in npf_i])
        npf_i_half_shifted = np.array([ray / np.linalg.norm(ray) for ray in npf_i_half_shifted])

        corrs = np.dot(npf_i_half_shifted, npf_i.T)  # no conjugation as the scl are conjugate-equal, not equal
        corrs = np.reshape(corrs, (n_shifts, n_theta//2, n_theta))
        corrs = np.array([corr*good_diffs for corr in corrs])  # mask with allowed combinations
        shift, scl1, scl2 = np.unravel_index(np.argmax(np.real(corrs)), corrs.shape)
        sclmatrix[i] = [scl1, scl2]
        corrs_stats[i] = np.real(corrs[(shift, scl1, scl2)])
        shifts_stats[i] = shift

    return sclmatrix, corrs_stats, shifts_stats


def estimate_relative_viewing_directions_cn(npf, cache_file_name=None, rots_gt=None):

    n_symm = AbinitioSymmConfig.n_symm
    if cache_file_name is None:
        base_dir = "."
        n_points_sphere = 1000
        n_theta = 360
        inplane_rot_res = 1
        cache_file_name, *_ = create_cache(base_dir, n_points_sphere, n_theta, inplane_rot_res, rots_gt)
    print('loading line indexes cache %s.\n Please be patient...' % cache_file_name)
    cijs_inds, Ris_tilde, R_theta_ijs, n_theta = read_cache(cache_file_name)
    AbinitioSymmConfig.n_theta = n_theta
    n_r = AbinitioSymmConfig.n_r
    if n_theta != npf.shape[1]:
        raise ValueError('n_theta = %d for cache, while n_theta=%d for npf are not equal. '
                         'Either create a new cache or a new npf' % (n_theta, len(npf)))
    print('done loading indexes cache')
    n_images = len(npf)
    n_cands = len(Ris_tilde)
    n_theta_ijs = len(R_theta_ijs)
    max_shift_1d = np.ceil(2 * np.sqrt(2) * AbinitioSymmConfig.max_shift)  # TODO extract this and put in ASC
    shift_phases = utils.calc_shift_phases(AbinitioSymmConfig.n_r, max_shift_1d, AbinitioSymmConfig.shift_step)
    n_shifts = len(shift_phases)
    # Step 1: pre-calculate the likelihood with respect to the self common-lines
    # note: cannot pre-compute the scls inds in cache since these depend on the symmetry class
    scls_inds = compute_scls_inds(Ris_tilde, n_symm, n_theta)
    scores_self_corrs = np.zeros((n_images, n_cands))
    for i in range(n_images):
        npf_i = npf[i]
        # generate all shifted versions of the images
        npf_i_half = npf_i[:n_theta // 2]

        npf_i_half_shifted = np.array([npf_i_half * shift_phase for shift_phase in shift_phases])  # shape is (n_shifts, n_theta/2, n_r)
        npf_i_half_shifted = np.reshape(npf_i_half_shifted, (-1, n_r))  # shape is (n_theta/2 * n_shifts, n_r)

        # ignoring dc-term.
        npf_i[:, 0] = 0
        npf_i_half_shifted[:, 0] = 0

        # normalize each ray to have norm equal to 1
        npf_i = np.array([ray / np.linalg.norm(ray) for ray in npf_i])
        npf_i_half_shifted = np.array([ray / np.linalg.norm(ray) for ray in npf_i_half_shifted])

        # TODO: there is no conjugation here because when building scl_inds we impliceteley picked
        #  the antipodal line. Contrast this with the c3_c4 algorithm. Suggestion: since in c3_c3 we
        #  utilize the upfront known angles between scls fix here rather than there
        corrs = np.dot(npf_i_half_shifted, np.conj(npf_i).T)
        corrs = np.reshape(corrs, (n_shifts, n_theta // 2, n_theta))
        corrs_cands = np.array([np.max(np.real(corrs[:, scls_inds_cand[:, 0], scls_inds_cand[:, 1]]), axis=0)
                                         for scls_inds_cand in scls_inds])
        scores_self_corrs[i] = np.mean(np.real(corrs_cands), axis=1)

    # removing candidates that are equator images
    cii_equators_inds = np.array([ind for (ind, Ri_tilde) in enumerate(Ris_tilde)
                                  if abs(np.arccos(Ri_tilde[2, 2]) - np.pi/2) < 10*np.pi/180])
    scores_self_corrs[:, cii_equators_inds] = 0

    n_theta_ijs_to_keep = (n_theta_ijs//n_symm)*n_symm
    if n_theta_ijs_to_keep < n_theta_ijs:
        cijs_inds = np.delete(cijs_inds, slice(n_theta_ijs_to_keep, cijs_inds.shape[2]), 2)
        R_theta_ijs = np.delete(R_theta_ijs, slice(n_theta_ijs_to_keep, R_theta_ijs.shape[0]), 0)
        print('number of inplane rotation angles must be divisible by n_symm')

    # Step 2: likelihood wrt to pairwise images
    print('computing pairwise likelihood')
    m_choose_2 = scipy.special.comb(n_images, 2).astype(int)
    vijs = np.zeros((m_choose_2, 3, 3))
    viis = np.zeros((n_images, 3, 3))
    g = utils.generate_g(n_symm)
    gs_s = np.array([np.linalg.matrix_power(g, s) for s in range(n_symm)])
    n_points_sphere, n_points_sphere, n_theta_ijs, _ = cijs_inds.shape
    c = 0
    e1 = [1, 0, 0]
    min_ii_norm = min_jj_norm = float('inf')
    for i in range(n_images):
        # generate all shifted versions of the images
        npf_i_half = npf[i, :n_theta // 2]

        npf_i_half_shifted = np.array([npf_i_half * shift_phase for shift_phase in shift_phases])  # shape is (n_shifts, n_theta/2, n_r)
        npf_i_half_shifted = np.reshape(npf_i_half_shifted, (-1, n_r))  # shape is (n_shifts * n_theta/2, n_r)
        # ignoring dc-term.
        npf_i_half_shifted[:, 0] = 0
        # normalize each ray to have norm equal to 1
        npf_i_half_shifted = np.array([ray / np.linalg.norm(ray) for ray in npf_i_half_shifted])

        for j in range(i+1, n_images):
            print(str(i), str(j))
            npf_j = npf[j]
            npf_j[:, 0] = 0
            # normalize each ray to have norm equal to 1
            npf_j = np.array([ray / np.linalg.norm(ray) for ray in npf_j])

            corrs_ij = np.dot(npf_i_half_shifted, np.conj(npf_j).T)
            # max out the shifts (recall each line may have its own shift)
            corrs_ij = np.max(np.reshape(np.real(corrs_ij), (n_shifts, n_theta//2, n_theta)), axis=0)
            corrs = corrs_ij[cijs_inds[..., 0], cijs_inds[..., 1]]
            corrs = np.reshape(corrs, (-1, n_symm, n_theta_ijs//n_symm))
            corrs = np.mean(corrs, axis=1)  # mean over all n_sym cls
            corrs = np.reshape(corrs, (n_points_sphere, n_points_sphere, n_theta_ijs//n_symm))
            #  the self common-lines are invariant to n_theta_ijs (i.e., in-plane rotation angles) so max them out
            opt_theta_ij_ind_per_sphere_points = np.argmax(corrs, axis=-1)
            corrs = np.max(corrs, axis=-1)
            # maximum likelihood while taking into consideration both cls and scls
            corrs = corrs * np.outer(scores_self_corrs[i], scores_self_corrs[j])

            opt_sphere_i, opt_sphere_j = np.unravel_index(np.argmax(corrs), corrs.shape)
            opt_theta_ij = opt_theta_ij_ind_per_sphere_points[opt_sphere_i, opt_sphere_j]

            opt_Ri_tilde = Ris_tilde[opt_sphere_i]
            opt_Rj_tilde = Ris_tilde[opt_sphere_j]
            opt_R_theta_ij = R_theta_ijs[opt_theta_ij]

            vii_j = np.mean(np.array([np.linalg.multi_dot([opt_Ri_tilde.T, gs, opt_Ri_tilde])
                                      for gs in gs_s]), axis=0)
            _, svals, _ = np.linalg.svd(vii_j)
            if np.linalg.norm(svals - e1, 2) < min_ii_norm:
                viis[i] = vii_j

            vjj_i = np.mean(np.array([np.linalg.multi_dot([opt_Rj_tilde.T, gs, opt_Rj_tilde])
                                      for gs in gs_s]), axis=0)
            _, svals, _ = np.linalg.svd(vjj_i)
            if np.linalg.norm(svals - e1, 2) < min_jj_norm:
                viis[j] = vjj_i

            vijs[c] = np.mean(np.array([np.linalg.multi_dot([opt_Ri_tilde.T, gs, opt_R_theta_ij, opt_Rj_tilde])
                                    for gs in gs_s]), axis=0)
            c += 1
    return viis, vijs


def compute_scls_inds(Ris_tilde, n_symm, n_theta):

    n_selfcl_pairs = (n_symm-1)//2
    n_cands = len(Ris_tilde)
    scls_inds = np.zeros((n_cands, n_selfcl_pairs, 2), dtype=np.uint16)
    g = utils.generate_g(n_symm)
    gs_s = np.array([np.linalg.matrix_power(g, s) for s in range(1, n_selfcl_pairs+1)])

    for i_cand in range(n_cands):
        Ri_tilde = Ris_tilde[i_cand]
        Riigs = np.array([np.linalg.multi_dot([Ri_tilde.T, gs, Ri_tilde]) for gs in gs_s])

        c1s = np.array([[-Riig[1, 2],  Riig[0, 2]] for Riig in Riigs])
        c2s = np.array([[ Riig[2, 1], -Riig[2, 0]] for Riig in Riigs])

        c1s_inds = utils.clAngles2Ind__(c1s, n_theta)
        c2s_inds = utils.clAngles2Ind__(c2s, n_theta)

        inds = np.where(c1s_inds >= (n_theta//2))
        c1s_inds[inds] -= (n_theta//2)
        c2s_inds[inds] += (n_theta//2)
        c2s_inds[inds] = np.mod(c2s_inds[inds], n_theta)

        scls_inds[i_cand, :, 0] = c1s_inds
        scls_inds[i_cand, :, 1] = c2s_inds
    return scls_inds


def create_cache(base_dir, n_points_sphere=1000, n_theta=360, inplane_rot_res_deg=1, rots_gt=None):
    print('creating cache')
    Ris_tilde, R_theta_ijs = generate_cand_rots(n_points_sphere, inplane_rot_res_deg, rots_gt)
    cijs_inds = compute_cls_inds(Ris_tilde, R_theta_ijs, n_theta)

    filename = "cn_cache_points%d_ntheta%d_res%d.pckl" % (n_points_sphere, n_theta, inplane_rot_res_deg)
    cache_mat_full_file_name = os.path.join(base_dir, filename)
    print('Saving data to cache file=%s' % cache_mat_full_file_name)
    f = open(cache_mat_full_file_name, 'wb')
    pickle.dump((cijs_inds, Ris_tilde, R_theta_ijs, n_theta), f)
    f.close()
    print('Cache file has been successfully saved!')
    return cache_mat_full_file_name, Ris_tilde, R_theta_ijs, cijs_inds


def generate_cand_rots(n_points_sphere, inplane_rot_res_deg, rots_gt=None):
    # Step 1: construct Ris_tilde (i.e. Ris modulo the in-plane rotation matrices)
    vis = generate_cand_rots_third_rows(n_points_sphere, rots_gt)
    Ris_tilde = np.array([utils.complete_third_row_to_rot(vi) for vi in vis])
    theta_ij = np.arange(0, 360, inplane_rot_res_deg) * np.pi / 180

    # Step 2: construct all in-plane rotation matrices R(theta_ij)
    cos_theta_ijs = np.cos(theta_ij)
    sin_theta_ijs = np.sin(theta_ij)
    R_theta_ijs = np.array([[[c, -s, 0], [s, c, 0], [0, 0, 1]] for (c, s) in zip(cos_theta_ijs, sin_theta_ijs)])

    return Ris_tilde, R_theta_ijs


def generate_cand_rots_third_rows(n_points_sphere, rots_gt=None):
    if rots_gt is None:
        third_rows = np.random.randn(n_points_sphere, 3)
        third_rows = np.array([third_row / np.linalg.norm(third_row) for third_row in third_rows])
    else:
        third_rows_gt = np.array([rot[-1] for rot in rots_gt])
        third_rows_other = np.random.randn(n_points_sphere - len(third_rows_gt), 3)
        third_rows_other = np.array([third_row / np.linalg.norm(third_row) for third_row in third_rows_other])
        third_rows = np.vstack((third_rows_gt, third_rows_other))
    return third_rows


def compute_cls_inds(Ris_tilde, R_theta_ijs, n_theta):
    n_points_sphere = len(Ris_tilde)
    n_theta_ijs = len(R_theta_ijs)
    cij_inds = np.zeros([n_points_sphere, n_points_sphere, n_theta_ijs, 2], dtype=np.uint16)

    with tqdm(total=n_points_sphere) as pbar:
        for i in range(n_points_sphere):
            for j in range(n_points_sphere):
                print(str(i), str(j))
                R_cands = np.array([np.linalg.multi_dot([Ris_tilde[i].T, R_theta_ij, Ris_tilde[j]])
                                    for R_theta_ij in R_theta_ijs])

                c1s = np.array([(-R_cand[1, 2], R_cand[0, 2]) for R_cand in R_cands])
                c2s = np.array([(R_cand[2, 1], -R_cand[2, 0]) for R_cand in R_cands])

                c1s = utils.clAngles2Ind__(c1s, n_theta)
                c2s = utils.clAngles2Ind__(c2s, n_theta)

                inds = np.where(c1s >= n_theta // 2)
                c1s[inds] -= n_theta // 2
                c2s[inds] += n_theta // 2
                c2s[inds] = np.mod(c2s[inds], n_theta)

                cij_inds[i, j, :, 0] = c1s
                cij_inds[i, j, :, 1] = c2s
            # update the bar
            if np.mod(i, 10) == 0:
                pbar.update(10)
    return cij_inds


def read_cache(cache_mat_full_file_name):
    f = open(cache_mat_full_file_name, 'rb')
    cijs_inds, Ris_tilde, R_theta_ijs, n_theta = pickle.load(f)
    f.close()
    print('Cache file has been successfully loaded!')
    return cijs_inds, Ris_tilde, R_theta_ijs, n_theta


def test_local_handedness_sync(n_symm, n_images=100):
    rots_gt = utils.generate_rots(n_images, is_J_conj_random=False)

    viis_gt = np.zeros((n_images, 3, 3))
    for i, Ri in enumerate(rots_gt):
        vi = Ri[2]
        viis_gt[i] = np.outer(vi, vi)

    vijs_gt = np.zeros((scipy.special.comb(n_images, 2).astype(int), 3, 3))
    c = 0  # counter
    for i in np.arange(n_images):
        for j in np.arange(i + 1, n_images):
            vi = rots_gt[i, 2]
            vj = rots_gt[j, 2]
            vijs_gt[c] = np.outer(vi, vj)
            c = c + 1

    g = utils.generate_g(n_symm)
    Riis = [np.linalg.multi_dot([Ri.T, g, Ri]) for Ri in rots_gt]
    for i, Rii in enumerate(Riis):
        if np.random.random() > 0.5:
            Rii = utils.J_conjugate(Rii)
        if np.random.random() > 0.5:
            Rii = Rii.T
        Riis[i] = Rii

    Rijs = np.zeros((scipy.special.comb(n_images, 2).astype(int), 3, 3))
    c = 0
    for i in np.arange(n_images):
        for j in np.arange(i + 1, n_images):
            Ri = rots_gt[i]
            Rj = rots_gt[j]
            s_ij = np.random.randint(n_symm)
            g_s_ij = np.linalg.matrix_power(g, s_ij)
            Rij = np.linalg.multi_dot([Ri.T, g_s_ij, Rj])
            if np.random.random() > 0.5:
                Rij = utils.J_conjugate(Rij)
            Rijs[c] = Rij
            c = c + 1

    viis, vijs = local_handedness_sync(n_symm, Riis, Rijs, rots_gt)

    diff_viis = np.zeros(n_images)
    for i, (vii, vii_gt) in enumerate(zip(viis, viis_gt)):
        diff_viis[i] = np.min([np.linalg.norm(vii - vii_gt, 'fro'), np.linalg.norm(utils.J_conjugate(vii) - vii_gt, 'fro')])
    mse_viis = np.sum(diff_viis ** 2) / n_images
    print("mse_viis=" + str(mse_viis))

    diff_vijs = np.zeros_like(vijs)
    for c, (vij, vij_gt) in enumerate(zip(vijs, vijs_gt)):
        diff_vijs[c] = np.min([np.linalg.norm(vij - vij_gt, 'fro'), np.linalg.norm(utils.J_conjugate(vij) - vij_gt, 'fro')])
    mse_vijs = np.sum(diff_vijs ** 2) / scipy.special.comb(n_images, 2)
    print("mse_vijs=" + str(mse_vijs))


def test_estimate_self_relative_rots(n_symm, n_theta, rots_gt):
    Riis = estimate_self_relative_rots(n_symm, n_theta, rots_gt)
    utils.detection_rate_self_relative_rots(Riis, n_symm, rots_gt)


def test_estimate_relative_rots(n_symm, n_theta, rots_gt):
    Rijs = utils.estimate_relative_rots(n_symm, n_theta, rots_gt)
    utils.detection_rate_relative_rots(Rijs, n_symm, rots_gt)


if __name__ == "__main__":
    rots_gt = utils.generate_rots(n_images=100, is_J_conj_random=True)
    test_estimate_self_relative_rots(n_symm=3, rots_gt=rots_gt, n_theta=360)
    test_estimate_relative_rots(n_symm=3, rots_gt=rots_gt, n_theta=360)
    # test_local_handedness_sync(n_symm=3, n_images = 100)
    # test_local_handedness_sync(n_symm=4, n_images = 100)