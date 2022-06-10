import os
import os.path as osp


def _get_env_var(name):
    if name not in os.environ:
        raise EnvironmentError("Required environment variable '{}' is not set".format(name))
    return os.environ[name]


class Paths(object):
    def __init__(self):
        raise ValueError("Static class 'Paths' should be not be initialized")

    @classmethod
    def configs_dir(cls):
        return osp.realpath(osp.join(osp.dirname(__file__), os.pardir, os.pardir, "configs"))

    @classmethod
    def workspace_dir(cls):
        return osp.join(_get_env_var("HODOR_WORKSPACE_DIR"))

    @classmethod
    def json_annotations_dir(cls):
        return osp.join(cls.workspace_dir(), "dataset_json_annotations")

    @classmethod
    def dataset_images_dir(cls):
        return osp.join(cls.workspace_dir(), "dataset_images")

    @classmethod
    def saved_models(cls):
        return osp.join(cls.workspace_dir(), "checkpoints")

    @classmethod
    def pretrained_backbones_dir(cls):
        return osp.join(cls.workspace_dir(), "pretrained_backbones")

    @classmethod
    def coco_train_images(cls):
        return osp.join(cls.dataset_images_dir(), "coco_2017_train")

    @classmethod
    def coco_train_anns(cls):
        return osp.join(cls.json_annotations_dir(), "coco_2017_train.json")

    @classmethod
    def davis_trainval_images_dir(cls):
        return osp.join(cls.dataset_images_dir(), "davis_trainval")

    @classmethod
    def davis_train_anns(cls):
        return osp.join(cls.json_annotations_dir(), "davis_train.json")

    @classmethod
    def davis_val_anns(cls):
        return osp.join(cls.json_annotations_dir(), "davis_val.json")

    @classmethod
    def davis_testdev_images_dir(cls):
        return osp.join(cls.dataset_images_dir(), "davis_testdev")

    @classmethod
    def davis_testdev_anns(cls):
        return osp.join(cls.json_annotations_dir(), "davis_testdev.json")

    @classmethod
    def youtube_vos_train_images_dir(cls):
        return osp.join(cls.dataset_images_dir(), "youtube_vos_2019_train")

    @classmethod
    def youtube_vos_train_anns(cls):
        return osp.join(cls.json_annotations_dir(), "youtube_vos_2019_train.json")

    @classmethod
    def youtube_vos_val_images_dir(cls):
        return osp.join(cls.dataset_images_dir(), "youtube_vos_2019_val")

    @classmethod
    def youtube_vos_val_anns(cls):
        return osp.join(cls.json_annotations_dir(), "youtube_vos_2019_val.json")
