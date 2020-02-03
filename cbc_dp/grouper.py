"""
grouper.py - rotational scan diffraction streaks grouping algorithm
"""
import numpy as np

class RotationGroup():
    """
    Rotational scan group class

    idxs - tilt series frames indices
    streaks - grouped diffraction streaks
    tolerance - streaks grouping tolerance
    """
    def __init__(self, frame_idxs, idxs, vec_list, tolerance):
        self.idxs, self.tol = dict(zip(frame_idxs, idxs)), tolerance
        self.strks = {frame_idx: vec_list[frame_idx][idx]
                      for frame_idx, idx in zip(frame_idxs, idxs)}
        self.strks_mean = {frame_idx: streaks.mean(axis=0)
                           for frame_idx, streaks in self.strks.items()
                           if streaks.any()}
        l_idx, l_strk = list(self.strks_mean.items())[-1]
        f_idx, f_strk = list(self.strks_mean.items())[-2]
        self.dist = (l_strk - f_strk) / (l_idx - f_idx)

    def __str__(self):
        return str(self.idxs)

    @property
    def streaks(self):
        return list(self.strks.values())

    @property
    def frame_idxs(self):
        return list(self.strks)

    @property
    def streaks_mean(self):
        return list(self.strks_mean.values())

    @property
    def last_idx(self):
        return list(self.strks_mean)[-1]

    @property
    def finished(self):
        return not (self.streaks[-2].any() or self.streaks[-1].any())

    def get_idxs(self, frame_idx):
        """
        Return streaks of the given frame index
        """
        return self.idxs.get(frame_idx, np.array([]))

    def search_frame(self, vec_list, frame_idx):
        """
        Search frame for a streak to group

        vec_list - tilt series
        frame_idx - frame index
        """
        new_vec = self.streaks_mean[-1] + (frame_idx - self.last_idx) * self.dist
        dist = np.sqrt(np.sum((vec_list[frame_idx] - new_vec)**2, axis=-1))
        idxs = np.where(dist < self.tol)[0]
        return self.append(frame_idx, idxs, vec_list)

    def append(self, frame_idx, idxs, vec_list):
        """
        Append streaks of a frame to the group

        frame_idx - frame index
        idxs - streak indices
        vec_list - tilt series
        """
        return RotationGroup(frame_idxs=self.frame_idxs + [frame_idx],
                             idxs=list(self.idxs.values()) + [idxs],
                             vec_list=vec_list,
                             tolerance=self.tol)

    def index_streak(self):
        """
        Return a streak for indexing
        """
        return np.mean(self.streaks_mean, axis=0)

class Grouper():
    """
    Grouping algorithm class

    vec_list - tilt series
    threshold - grouping threshold
    """
    def __init__(self, vec_list, threshold):
        self.threshold = threshold
        self._init_streaks(vec_list)
        self.finished_groups = []
        self._process_vec(vec_list)

    @property
    def groups(self):
        return self.finished_groups + self.ongoing_groups

    def _init_streaks(self, vec_list):
        pairs = self.find_pairs(vec_list, 0)
        groups = self.group_pairs(pairs)
        self.ongoing_groups = [RotationGroup(frame_idxs=[0, 1],
                                             idxs=group,
                                             vec_list=vec_list,
                                             tolerance=self.threshold / 2) for group in groups]

    def _update_groups(self, vec_list, idx):
        ongoing_groups = []
        for group in self.ongoing_groups:
            group = group.search_frame(vec_list, idx)
            if group.finished:
                self.finished_groups.append(group)
            else:
                ongoing_groups.append(group)
        self.ongoing_groups = ongoing_groups

    def _process_vec(self, vec_list):
        for idx in range(1, len(vec_list) - 1):
            self._update_groups(vec_list, idx + 1)
            pairs = self.find_pairs(vec_list, idx)
            pairs = self.filter_pairs(pairs, idx)
            groups = self.group_pairs(pairs)
            new_groups = [RotationGroup(frame_idxs=[idx, idx + 1],
                                        idxs=group,
                                        vec_list=vec_list,
                                        tolerance=self.threshold / 2) for group in groups]
            self.ongoing_groups.extend(new_groups)

    def ongoing_streaks(self, idx):
        """
        Return ongoing groups streaks of the given frame

        idx - frame index
        """
        idxs = [group.get_idxs(idx)
                for group in self.ongoing_groups]
        return np.concatenate(idxs)

    def find_pairs(self, vec_list, idx):
        """
        Find pairs to group two adjacent frames with indices [idx, idx + 1]

        vec_list - tilt series
        idx - frame index
        """
        dist = np.sqrt(np.sum((vec_list[idx][:, None, :] - vec_list[idx + 1][None, :, :])**2,
                              axis=-1))
        idxs = np.where(dist < self.threshold)
        return np.stack(idxs, axis=1)

    def filter_pairs(self, pairs, idx):
        """
        Filter pairs of two adjacent frames from already found rotational groups

        pairs - pairs to filter
        idx - frame index
        """
        ong_strks = self.ongoing_streaks(idx)
        pairs_mask = (pairs[:, None, 0] != ong_strks[None, :]).all(axis=1)
        return pairs[pairs_mask]

    def group_pairs(self, pairs):
        """
        Group pairs of two adjacent frames into groups

        pairs - pairs to group
        """
        groups = []
        for pair in pairs:
            for idx, group in enumerate(groups):
                if (pair[:, None] == group).any():
                    groups[idx] = np.concatenate([group, pair[:, None]], axis=1)
                    break
            else:
                groups.append(pair[:, None])
        return groups

class TiltGroups():
    """
    Tilt series rotational groups class

    vec_list - tilt series
    threshold - grouping threshold
    """
    def __init__(self, kout, kin, threshold):
        self.kout, self.kin = kout, kin
        self.groups = Grouper(self.scat_vec, threshold).groups

    @property
    def scat_vec(self):
        return [kout -  kin for kout, kin in zip(self.kout, self.kin)]

    def frame_groups(self, frame_idx):
        """
        Return rotational groups streaks of the given frame
        """
        groups = [group for group in self.groups if group.get_idxs(frame_idx).any()]
        return groups

    def frame_streaks(self, frame_idx):
        """
        Return refined rotational groups streaks of the given frame

        idx - frame index
        """
        idxs, streaks = [], []
        for group in self.frame_groups(frame_idx):
            group_idxs = group.get_idxs(frame_idx)
            idxs.append(group_idxs)
            streaks.append(np.tile(group.index_streak()[None, :], (group_idxs.size, 1)))
        return np.concatenate(idxs), np.concatenate(streaks)

    def ref_kin(self):
        """
        Returned tilt series incoming wavevectors with rotational groups collapsed
        """
        kin_list = []
        for idx, (kout, kin) in enumerate(zip(self.kout, self.kin)):
            frame_groups = self.frame_groups(idx)
            if frame_groups:
                idxs, streaks = self.frame_streaks(idx)
                frame_kin = np.copy(kin)
                frame_kin[idxs] = kout[idxs] - streaks
                kin_list.append(frame_kin)
            else:
                kin_list.append(kin)
        return kin_list
