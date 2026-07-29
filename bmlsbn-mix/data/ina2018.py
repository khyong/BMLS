import os

from data.long_tailed_dataset import LongTailedImageDataset


class iNa2018(LongTailedImageDataset):
    num_classes = 8142
    cls_num = num_classes
    subset_thr = [100, 20]

    def __init__(
        self,
        root,
        args,
        train=True,
        batch_size=256,
        transform=None,
        strong_transform=None,
        target_transform=None,
        class_map=None,
        list_file=None,
    ):
        if list_file is None:
            list_file = self._find_list_file(root, train)
        super(iNa2018, self).__init__(
            root=root,
            list_file=list_file,
            args=args,
            num_classes=self.num_classes,
            train=train,
            batch_size=batch_size,
            transform=transform,
            strong_transform=strong_transform,
            target_transform=target_transform,
            class_map=class_map,
            subset_thr=self.subset_thr,
        )

    @staticmethod
    def _find_list_file(root, train):
        split = 'train' if train else 'val'
        filenames = [
            'iNaturalist18_{}.txt'.format(split),
            'iNaturalist2018_{}.txt'.format(split),
            '{}.txt'.format(split),
        ]
        candidates = []
        list_root = os.environ.get('BMLSBN_INAT_LIST_ROOT')
        if list_root:
            candidates.extend([
                os.path.join(os.path.expanduser(list_root), filename)
                for filename in filenames
            ])
        for filename in filenames:
            candidates.extend([
                os.path.join(root, filename),
                os.path.join(root, 'annotations', filename),
                os.path.join(root, 'iNaturalist18', filename),
                os.path.join(os.path.dirname(root), 'data_txt', filename),
                os.path.join(os.path.dirname(__file__), 'iNaturalist18', filename),
            ])
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            "Could not find a {} list file. Checked: {}".format(
                split, ', '.join(candidates)
            )
        )
