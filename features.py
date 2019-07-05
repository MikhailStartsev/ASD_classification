import numpy as np
from scipy import misc, ndimage, stats
import json


def check_intersection(data_pt, rects, inflate_factor=1.5):
    half_addition = (inflate_factor - 1.0) / 2.0
    for r in rects:
        w, h = r['right'] - r['left'], r['bottom'] - r['top']

        if r['left'] - w * half_addition <= data_pt['x'] <= r['right'] + half_addition and \
                                        r['top'] - h * half_addition <= data_pt['y'] <= r['bottom'] + h * half_addition:
            return True

    return False


def extract_features(all_data, image_path, SAM_saliency_path,
                     face_rectangles_json_path=None,  # will not be used if not provided
                     return_names=False,
                     extra_blur=True,
                     debug=False):
    restore_to_one_element = False
    if not isinstance(all_data, list):
        all_data = [all_data]
        restore_to_one_element = True

    overall_result = []
    overall_names = []

    image_size = misc.imread(image_path).shape[:2]

    loaded_saliency_maps = {
        'SAM': misc.imread(SAM_saliency_path).astype(np.float)
    }
    if extra_blur:
        for k in loaded_saliency_maps:
            # sigma ~= 1 deg
            ndimage.gaussian_filter(loaded_saliency_maps[k], sigma=43, output=loaded_saliency_maps[k])

    rectangle_inflation_coeff = 1.5
    if face_rectangles_json_path is not None:
        face_rects = json.load(open(face_rectangles_json_path))
        for r in face_rects:
            r['area_share'] = (r['right'] - r['left']) * (r['bottom'] - r['top']) * (rectangle_inflation_coeff ** 2) \
                              / float(image_size[0] * image_size[1])
        total_rect_area_share = sum([r['area_share'] for r in face_rects])

    for data in all_data:
        res = []
        res_names = []

        # total and mean duration
        res.append(len(data))
        res_names.append('scanpath_fix_count')

        # total and mean duration
        res.append(np.sum(data['duration']))
        res_names.append('scanpath_fix_duration_ms_total')

        res.append(np.mean(data['duration']))
        res_names.append('scanpath_fix_duration_ms_mean')

        # total, mean, and median saccade amplitude
        differences = np.diff(data[['x', 'y']].tolist(), axis=0)
        amplitudes = np.linalg.norm(differences, axis=1)
        assert len(amplitudes) == len(data) - 1

        res.append(np.sum(amplitudes))
        res_names.append('scanpath_len_px_total')

        res.append(np.mean(amplitudes) if len(amplitudes) else 0.0)
        res_names.append('scanpath_saccade_amplitude_px_mean')

        # mean and median fixation distance to centre
        distance_to_centre = np.sqrt((data['x'] - image_size[1] / 2) ** 2 +
                                     (data['y'] - image_size[0] / 2) ** 2)
        res.append(np.mean(distance_to_centre))
        res_names.append('scanpath_distance_to_centre_px_mean')

        mean_x, mean_y = data['x'].mean(), data['y'].mean()
        distance_to_mean = np.sqrt((data['x'] - mean_x) ** 2 +
                                   (data['y'] - mean_y) ** 2)
        res.append(np.mean(distance_to_mean))
        res_names.append('scanpath_distance_to_scanpath_mean_px_mean')

        # face features
        if face_rectangles_json_path is not None:
            # applicable for 1+ faces
            # percentage of fixations on faces + time spent on faces + percentage of this time
            if len(face_rects):
                num_on_faces = 0.0
                dur_on_faces = 0.0
                for p in data:
                    if check_intersection(p, face_rects, inflate_factor=rectangle_inflation_coeff):
                        num_on_faces += 1
                        dur_on_faces += p['duration']

                res.append(num_on_faces)
                res.append(num_on_faces / len(data))
                res.append(num_on_faces / total_rect_area_share)
                res.append(num_on_faces / len(data) / total_rect_area_share)
                res.append(dur_on_faces)
                res.append(dur_on_faces / np.sum(data['duration']))
                res.append(dur_on_faces / total_rect_area_share)
                res.append(dur_on_faces / np.sum(data['duration']) / total_rect_area_share)
            else:
                res += [np.nan] * 8
            res_names += ['faces_fix_count', 'faces_fix_share',
                          'faces_fix_count_normalised_by_area', 'faces_fix_share_normalised_by_area',
                          'faces_fix_duration_ms', 'faces_fix_duration_share',
                          'faces_fix_duration_ms_normalised_by_area', 'faces_fix_duration_share_normalised_by_area']

            # applicable for 2+ faces
            if len(face_rects) >= 2:
                percentages_on_faces = np.zeros(len(face_rects))
                percentages_on_faces_normalised = np.zeros(len(face_rects))

                time_on_faces = np.zeros(len(face_rects))
                time_on_faces_normalised = np.zeros(len(face_rects))
                for p in data:
                    for rect_i, r in enumerate(face_rects):
                        if check_intersection(p, [r]):
                            percentages_on_faces[rect_i] += 1
                            time_on_faces[rect_i] += p['duration']

                for rect_i, r in enumerate(face_rects):
                    percentages_on_faces_normalised[rect_i] = percentages_on_faces[rect_i] / r['area_share']
                    time_on_faces_normalised[rect_i] = time_on_faces[rect_i] / r['area_share']

                if percentages_on_faces.sum() != 0:
                    percentages_on_faces /= percentages_on_faces.sum()
                    time_on_faces /= time_on_faces.sum()

                    res.append(stats.entropy(percentages_on_faces))
                    res.append(stats.entropy(time_on_faces))
                    res.append(stats.entropy(percentages_on_faces_normalised))
                    res.append(stats.entropy(time_on_faces_normalised))
                else:
                    res += [0.0] * 4
            else:
                res += [np.nan] * 4
            res_names += ['multi_face_fixation_count_entropy', 'multi_face_fixation_duration_entropy',
                          'multi_face_fixation_count_normalised_by_area_entropy', 'multi_face_fixation_duration_normalised_by_area_entropy']
            if debug:
                res_names += ['multi_face_fixation_count_deviation_from_unitform_max', 'multi_face_fixation_duration_deviation_from_unitform_max',
                              'multi_face_fixation_count_deviation_from_unitform_mean', 'multi_face_fixation_duration_deviation_from_unitform_mean']

        empirical_saliency = np.zeros(image_size)
        empirical_saliency[(data['y'].astype(int) - 1, data['x'].astype(int) - 1)] = 1
        ndimage.gaussian_filter(empirical_saliency, sigma=43, output=empirical_saliency)

        for smap_name in sorted(loaded_saliency_maps.keys()):
            sal_map = loaded_saliency_maps[smap_name].copy()
            sal_values = sal_map[(data['y'].astype(int) - 1, data['x'].astype(int) - 1)]

            res.append(sal_values[0])
            res_names.append('saliency_{}_first_fixation'.format(smap_name))

            max_val = sal_map.max()
            for max_share in [0.75, 0.9]:
                try:
                    first_reaching = np.where(sal_values >= max_val * max_share)[0][0] + 1
                except IndexError:
                    first_reaching = max(20, len(sal_values) + 1)
                res.append(first_reaching)
                res_names.append('saliency_{}_first_above_{}*max_rank'.format(smap_name, max_share))

            res.append(np.mean(sal_values))
            res_names.append('saliency_{}_mean'.format(smap_name))

            res.append(np.sum(sal_values))
            res_names.append('saliency_{}_sum'.format(smap_name))

            res.append(np.sum(data['duration'] * sal_values))
            res_names.append('saliency_{}_weighted_duration_sum'.format(smap_name))

            res.append(np.mean(data['duration'] * sal_values))
            res_names.append('saliency_{}_weighted_duration_mean'.format(smap_name))

            res.append(np.max(sal_values))
            res_names.append('saliency_{}_max'.format(smap_name))\

            # normalisation for KLD
            empirical_saliency -= empirical_saliency.min()
            empirical_saliency /= empirical_saliency.sum()
            sal_map -= sal_map.min()
            if sal_map.sum() == 0:
                sal_map = np.ones_like(sal_map)
            sal_map /= sal_map.sum()

            eps = np.finfo(sal_map.dtype).eps
            res.append((empirical_saliency * np.log(empirical_saliency / (sal_map + eps) + eps)).sum())
            res_names.append('saliency_{}_KLD'.format(smap_name))

            # normalisation for NSS
            sal_map -= sal_map.mean()
            if sal_map.std() != 0:
                sal_map /= sal_map.std()

            res.append(np.mean(sal_map[(data['y'].astype(int) - 1, data['x'].astype(int) - 1)]))
            res_names.append('saliency_{}_NSS'.format(smap_name))

        overall_result.append(res)
        assert len(res) == len(res_names)
        if not overall_names:
            overall_names = res_names

    if restore_to_one_element:
        overall_result = overall_result[0]

    if not return_names:
        return overall_result
    return overall_result, res_names