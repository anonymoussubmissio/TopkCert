import numpy as np


def malicious_list_drs(prediction_map_drs, ablation_size, patch_size, num_classses=1000):
    malicious_list = []
    delta = ablation_size + patch_size - 1
    pred_list, cnt = np.unique(prediction_map_drs, return_counts=True)
    sorted_idx = np.argsort(cnt)
    majority_pred = pred_list[sorted_idx][-1]
    sorted_value = np.sort(cnt)
    max_vote_value = sorted_value[-1]
    for idx in range(len(cnt)):
        if abs(cnt[idx] - max_vote_value) <= 2 * delta:
            malicious_list.append(pred_list[idx])
    if max_vote_value <= 2 * delta:
        for label in range(num_classses):
            malicious_list.append(label)
    return malicious_list


def certified_drs_four_delta(prediction_map_drs, ablation_size, patch_size):
    delta = ablation_size + patch_size - 1
    pred_list, cnt = np.unique(prediction_map_drs, return_counts=True)
    sorted_idx = np.argsort(cnt)
    majority_pred = pred_list[sorted_idx][-1]
    sorted_value = np.sort(cnt)
    # get majority prediction and disagreer prediction
    if len(sorted_value) > 1:
        gap = sorted_value[-1] - sorted_value[-2]
    else:
        gap = sorted_value[-1]
    if gap > 4 * delta:
        return majority_pred, True
    else:
        return majority_pred, False


def certified_drs_three_position(prediction_map_drs, ablation_size, patch_size):
    delta = ablation_size + patch_size - 1
    pred_list, cnt = np.unique(prediction_map_drs, return_counts=True)
    sorted_idx = np.argsort(cnt)
    majority_pred = pred_list[sorted_idx][-1]
    sorted_value = np.sort(cnt)
    # get majority prediction and disagreer prediction
    if len(sorted_value) > 2:
        gap = sorted_value[-1] - sorted_value[-3]
    else:
        gap = sorted_value[-1]
    if gap > 2 * delta:
        return majority_pred, True
    else:
        return majority_pred, False


def certified_drs_pg_version(total_num_class, prediction_map_drs, ablation_size, patch_size, label, image_size=224):
    worst_position=1
    # Stage 1: original votes
    delta = ablation_size + patch_size - 1
    pred_list, cnt = np.unique(prediction_map_drs, return_counts=True)
    sorted_idx = np.argsort(cnt)
    if label in pred_list:
        position_benign_no_order=np.where(pred_list==label)
        position_benign_order=np.where(sorted_idx==position_benign_no_order[0])
        position_benign_order=len(sorted_idx)-position_benign_order[0]-1
        position_benign_num=position_benign_order[0]+1
        position_benign_num=position_benign_num
    else:
        position_benign_num=total_num_class
    # majority_pred_ori = pred_list[sorted_idx][-1]
    majority_pred_ori=label
    majority_vote_ori = cnt[sorted_idx][-1]

    for i in range(image_size):
        worst_position_in_this_position=-1
        # Stage 2: clean votes
        # left right for the affect region, i is the attack region left
        if i + patch_size > image_size:
            break
        prediction_map_drs_tmp = prediction_map_drs.copy()
        left = i - ablation_size + 1
        right = i + patch_size
        if left < 0:
            left = image_size + left
            prediction_map_drs_clean = prediction_map_drs_tmp[right:left]
        else:
            front = prediction_map_drs_tmp[:left]
            behind = prediction_map_drs_tmp[right:]
            prediction_map_drs_clean = np.concatenate((front, behind), axis=0)
        # assertion
        if not len(prediction_map_drs_clean) == image_size - delta:
            print("wrong!")
        pred_list_test, cnt_test = np.unique(prediction_map_drs_clean, return_counts=True)

        # Stage 2.2: calculate the lower bound vote of the first label
        prediction_map_drs_clean_first_label = prediction_map_drs_clean[
            prediction_map_drs_clean == majority_pred_ori]
        pred_list_first, cnt_first = np.unique(prediction_map_drs_clean_first_label, return_counts=True)
        if len(cnt_first)==0:
            majority_vote_clean=0
        else:
            majority_vote_clean = cnt_first[-1]

        # Stage 3: clean votes without the first label
        prediction_map_drs_clean_without_first_label = prediction_map_drs_clean[
            prediction_map_drs_clean != majority_pred_ori]
        pred_list_clean_without_first, cnt_clean_without_first = np.unique(prediction_map_drs_clean_without_first_label,
                                                                           return_counts=True)
        # count
        sorted_idx_clean_without_first = np.argsort(cnt_clean_without_first)
        sorted_value_clean_without_first = np.sort(cnt_clean_without_first)
        # upper bound
        sorted_value_clean_without_first=sorted_value_clean_without_first+delta
        # where majority_vote_clean place?
        larger_or_eqaul_num=np.count_nonzero(sorted_value_clean_without_first>=majority_vote_clean)
        # whether run out
        if delta+0>=majority_vote_clean:
            worst_position_in_this_position = total_num_class
        else:
            worst_position_in_this_position=larger_or_eqaul_num+1

        # count in
        if worst_position_in_this_position>worst_position:
            worst_position=worst_position_in_this_position

    output_label_drs_new, worst_position_new = certified_drs_new_version(total_num_class, prediction_map_drs, ablation_size,
                                                                          patch_size, label)
    if worst_position<worst_position_new:
        output_label_drs_new, worst_position_new = certified_drs_new_version(total_num_class, prediction_map_drs, ablation_size,
                                                                             patch_size, label)
        output_label_drs_pg, robust_drs_pg_rank = certified_drs_pg_version(total_num_class, prediction_map_drs,
                                                                           ablation_size,
                                                                           patch_size, label)

    return position_benign_num, worst_position

def certified_drs_new_version(total_num_class, prediction_map_drs, ablation_size, patch_size, label, image_size=224):
    # Stage 1: original votes
    delta = ablation_size + patch_size - 1
    pred_list, cnt = np.unique(prediction_map_drs, return_counts=True)
    sorted_idx = np.argsort(cnt)
    # position_benign=np.where(sorted_idx==label)
    # position_benign_num=position_benign[0]+1
    if label in pred_list:
        position_benign_no_order=np.where(pred_list==label)
        position_benign_order=np.where(sorted_idx==position_benign_no_order[0])
        position_benign_order=len(sorted_idx)-position_benign_order[0]-1
        position_benign_num=position_benign_order[0]+1
        position_benign_num=position_benign_num

    else:
        position_benign_num=total_num_class
    majority_pred_ori=label
    # majority_pred_ori = pred_list[sorted_idx][-1]
    # majority_vote_ori = cnt[sorted_idx][-1]
    worst_position=1
    for i in range(image_size):
        score=delta
        # Stage 2: clean votes
        # left right for the affect region, i is the attack region left
        if i + patch_size > image_size:
            break
        prediction_map_drs_tmp = prediction_map_drs.copy()
        left = i - ablation_size + 1
        right = i + patch_size
        if left < 0:
            left = image_size + left
            prediction_map_drs_clean = prediction_map_drs_tmp[right:left]
        else:
            front = prediction_map_drs_tmp[:left]
            behind = prediction_map_drs_tmp[right:]
            prediction_map_drs_clean = np.concatenate((front, behind), axis=0)
        # assertion
        if not len(prediction_map_drs_clean) == image_size - delta:
            print("wrong!")
        pred_list_test, cnt_test = np.unique(prediction_map_drs_clean, return_counts=True)

        # Stage 2.2: calculate the lower bound vote of the first label
        prediction_map_drs_clean_first_label = prediction_map_drs_clean[
            prediction_map_drs_clean == majority_pred_ori]
        pred_list_first, cnt_first = np.unique(prediction_map_drs_clean_first_label, return_counts=True)
        if len(cnt_first)==0:
            majority_vote_clean=0
        else:
            majority_vote_clean = cnt_first[-1]

        # Stage 3: clean votes without the first label
        prediction_map_drs_clean_without_first_label = prediction_map_drs_clean[
            prediction_map_drs_clean != majority_pred_ori]
        pred_list_clean_without_first, cnt_clean_without_first = np.unique(
            prediction_map_drs_clean_without_first_label,
            return_counts=True)
        sorted_idx_clean_without_first = np.argsort(cnt_clean_without_first)
        sorted_value_clean_without_first = np.sort(cnt_clean_without_first)

        worst_position_in_this_position=1
        for cnt_idx in range(len(sorted_value_clean_without_first)):
            if sorted_value_clean_without_first[-(cnt_idx+1)]<majority_vote_clean:
                gap=majority_vote_clean-sorted_value_clean_without_first[-(cnt_idx+1)]
                score=score-gap
                if score<0:
                    worst_position_in_this_position=cnt_idx+1
                    break
                if score==0:
                    worst_position_in_this_position=cnt_idx+1+1
                    break
                if score>0:
                    worst_position_in_this_position=cnt_idx+1+1
            else:
                worst_position_in_this_position=cnt_idx+1+1
        if score>0:
            if majority_vote_clean>0:
                more_label=score/majority_vote_clean
            else:
                more_label=total_num_class
            worst_position_in_this_position=int(more_label)+worst_position_in_this_position
        if worst_position_in_this_position>worst_position:
            worst_position=worst_position_in_this_position
    if worst_position>total_num_class:
        worst_position=total_num_class
    return position_benign_num, worst_position

    #     sorted_idx_clean_without_first = np.argsort(cnt_clean_without_first)
    #     sorted_value_clean_without_first = np.sort(cnt_clean_without_first)
    #     if len(sorted_value_clean_without_first) >= k:
    #         if sorted_value_clean_without_first[-k] + delta >= majority_vote_clean:
    #             return majority_pred_ori, False
    #         else:
    #             continue
    #     else:
    #         if delta >= majority_vote_clean:
    #             return majority_pred_ori, False
    #         else:
    #             continue
    #
    # return majority_pred_ori, True

    # prediction_map_drs_clean=prediction_map_drs_tmp[]
    # pred_list, cnt = np.unique(prediction_map_drs, return_counts=True)
    # sorted_idx = np.argsort(cnt)
    # majority_pred = pred_list[sorted_idx][-1]
    # sorted_value = np.sort(cnt)
    # # get majority prediction and disagreer prediction
    # if len(sorted_value) > 2:
    #     gap = sorted_value[-1] - sorted_value[-3]
    # else:
    #     gap = sorted_value[-1]
    # if gap > 2 * delta:
    #     return majority_pred, True
    # else:
    #     return majority_pred, False


def certified_drs_three_position_mixup(prediction_map_drs_1, prediction_map_drs_2, ablation_size, patch_size):
    delta = ablation_size + patch_size - 1
    delta = delta * 2
    prediction_map_drs = np.concatenate((prediction_map_drs_1, prediction_map_drs_2), axis=0)
    pred_list, cnt = np.unique(prediction_map_drs, return_counts=True)
    sorted_idx = np.argsort(cnt)
    majority_pred = pred_list[sorted_idx][-1]
    sorted_value = np.sort(cnt)
    # get majority prediction and disagreer prediction
    if len(sorted_value) > 3:
        gap = sorted_value[-1] - sorted_value[-3]
    else:
        gap = sorted_value[-1]
    if gap > 2 * delta:
        return majority_pred, True
    else:
        return majority_pred, False
