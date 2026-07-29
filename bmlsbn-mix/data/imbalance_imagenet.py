import os

from data.long_tailed_dataset import LongTailedImageDataset


class ImageNetLT(LongTailedImageDataset):
    num_classes = 1000
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
        super(ImageNetLT, self).__init__(
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
        filenames = (
            ['ImageNet_LT_train.txt']
            if train
            else ['ImageNet_LT_val.txt', 'ImageNet_LT_test.txt']
        )
        candidates = []
        for filename in filenames:
            candidates.extend([
                os.path.join(root, filename),
                os.path.join(root, 'annotations', filename),
                os.path.join(root, 'ImageNet_LT', filename),
                os.path.join(os.path.dirname(root), 'data_txt', filename),
                os.path.join(os.path.dirname(__file__), 'ImageNet_LT', filename),
            ])
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            "Could not find an ImageNet-LT {} list. Checked: {}".format(
                'training' if train else 'validation/test',
                ', '.join(candidates),
            )
        )
