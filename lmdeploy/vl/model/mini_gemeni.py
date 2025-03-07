# Copyright (c) OpenMMLab. All rights reserved.

import os.path as osp
import warnings
from contextlib import contextmanager
from typing import Dict, List

import torch

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import add_device_hook, disable_logging, disable_transformers_logging, hack_import_with

logger = get_logger('lmdeploy')


def check_mini_gemini_install():
    """check mini gemini install."""
    try:
        with hack_import_with(['deepspeed']):
            import mgm  # noqa: F401
    except ImportError:
        raise ImportError('To use MiniGeminiVisionModel, please install minigemini by '
                          '`pip install git+https://github.com/dvlab-research/MGM.git'
                          ' --no-deps`')


def _build_vision_tower(vision_tower_cfg, **kwargs):

    from mgm.model.multimodal_encoder.builder import CLIPVisionTower, EVAVisionTower

    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    image_processor = getattr(vision_tower_cfg, 'image_processor',
                              getattr(vision_tower_cfg, 'image_processor', '../processor/clip-patch14-224'))

    if 'openai' in vision_tower.lower() or 'ShareGPT4V' in vision_tower:
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif 'lavis' in vision_tower.lower() or 'eva' in vision_tower.lower():
        return EVAVisionTower(vision_tower, image_processor, args=vision_tower_cfg, **kwargs)
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')


def _build_vision_tower_aux(vision_tower_cfg, **kwargs):

    from mgm.model.multimodal_encoder.builder import CLIPVisionTower, OpenCLIPVisionTower

    vision_tower_aux = getattr(vision_tower_cfg, 'mm_vision_tower_aux',
                               getattr(vision_tower_cfg, 'vision_tower_aux', None))

    if 'openclip' in vision_tower_aux.lower():
        return OpenCLIPVisionTower(vision_tower_aux, args=vision_tower_cfg, **kwargs)
    elif 'openai' in vision_tower_aux.lower():
        return CLIPVisionTower(vision_tower_aux, args=vision_tower_cfg, **kwargs)
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower_aux}')


def _clip_vision_tower__init__(self, vision_tower, args, delay_load=False):
    torch.nn.Module.__init__(self)

    self.is_loaded = False
    self.vision_tower_name = vision_tower if osp.exists(vision_tower) else 'openai/clip-vit-large-patch14-336'
    self.select_layer = args.mm_vision_select_layer
    self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
    self.is_optimize = getattr(args, 'optimize_vision_tower', False)


def _clip_vision_tower_load_model(self):
    from mgm.model.multimodal_encoder.clip_encoder import CLIPVisionModel, VideoFramesProcessor  # noqa
    self.image_processor = VideoFramesProcessor.from_pretrained(self.vision_tower_name)
    from transformers import CLIPVisionConfig
    config = CLIPVisionConfig.from_pretrained(self.vision_tower_name, trust_remote_code=True)
    self.vision_tower = CLIPVisionModel._from_config(config=config)
    self.vision_tower.requires_grad_(False)
    self.is_loaded = True


def _openclip_vision_tower__init__(self, vision_tower, args, delay_load=False):
    torch.nn.Module.__init__(self)
    self.is_loaded = False
    self.vision_tower_name = vision_tower
    if osp.exists(osp.join(vision_tower, 'open_clip_config.json')):
        import json
        self.vision_config = json.load(open(osp.join(vision_tower, 'open_clip_config.json'), 'r'))
    self.is_optimize = getattr(args, 'optimize_vision_tower_aux', False)


def _openclip_vision_tower_load_model(self):
    if 'convnext' in self.vision_tower_name:
        if 'large' in self.vision_tower_name and 'd-320' in self.vision_tower_name:  # noqa
            self.model_type = 'convnext_large_d_320'
            self.model_channel = [192, 384, 768, 1536]  # stage 0-3
        elif 'base' in self.vision_tower_name and 'w-320' in self.vision_tower_name:  # noqa
            self.model_type = 'convnext_base_w_320'
            self.model_channel = [128, 256, 512, 1024]
        elif 'xxlarge' in self.vision_tower_name:
            self.model_type = 'convnext_xxlarge'
            self.model_channel = [384, 768, 1536, 3072]

    from mgm.model.multimodal_encoder.openclip_encoder import CLIP, get_model_config  # noqa
    clip_model = CLIP(**get_model_config(self.model_type))
    clip_model.visual.trunk.norm_pre = None
    clip_model.visual.trunk.head = None
    clip_model.visual.head = None

    self.is_loaded = True
    # decompose stem and stages blocks in vision tower
    self.vision_stem = clip_model.visual.trunk.stem
    self.vision_stages = clip_model.visual.trunk.stages
    self.vision_stem.requires_grad_(False)
    self.vision_stages.requires_grad_(False)


old_func = torch.nn.Module.load_state_dict


def _load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
    return old_func(self, state_dict, strict=False, assign=assign)


@contextmanager
def init_mini_gemini_model():
    origin_func_path = [
        'mgm.model.multimodal_encoder.builder.build_vision_tower',
        'mgm.model.multimodal_encoder.builder.build_vision_tower_aux',
        'mgm.model.multimodal_encoder.clip_encoder.CLIPVisionTower.__init__',  # noqa: E501
        'mgm.model.multimodal_encoder.clip_encoder.CLIPVisionTower.load_model',  # noqa: E501
        'mgm.model.multimodal_encoder.openclip_encoder.OpenCLIPVisionTower.__init__',  # noqa: E501
        'mgm.model.multimodal_encoder.openclip_encoder.OpenCLIPVisionTower.load_model',  # noqa: E501
        'torch.nn.Module.load_state_dict',
    ]
    rewrite_func = [
        _build_vision_tower,
        _build_vision_tower_aux,
        _clip_vision_tower__init__,
        _clip_vision_tower_load_model,
        _openclip_vision_tower__init__,
        _openclip_vision_tower_load_model,
        _load_state_dict,
    ]
    from lmdeploy.vl.model.utils import rewrite_ctx
    with rewrite_ctx(origin_func_path, rewrite_func):
        yield


@VISION_MODELS.register_module()
class MiniGeminiVisionModel(VisonModel):
    """Qwen vision model."""

    _arch = ['MiniGeminiLlamaForCausalLM', 'MGMLlamaForCausalLM']

    def build_preprocessor(self):
        # pytorch engine will not support mini-gemini. Therefore, in order to
        # reuse the previous code as much as possible, we do not extract image
        # preprocessor from `build_model` function.
        pass

    def build_model(self):
        """build the vision part of a VLM model when backend is turbomind, or
        load the whole VLM model when `self.with_llm==True`"""
        check_mini_gemini_install()
        # empty init
        from accelerate import init_empty_weights
        from mgm.mm_utils import process_images
        from mgm.model import MGMLlamaForCausalLM  # noqa
        from mgm.model.language_model.mgm_llama import MGMConfig
        from transformers import AutoModelForCausalLM
        with init_empty_weights(), disable_transformers_logging(), hack_import_with(['deepspeed']):
            warnings.simplefilter('ignore')
            with init_mini_gemini_model():
                config = MGMConfig.from_pretrained(self.model_path, trust_remote_code=True)
                setattr(config, 'quantization_config', {})
                setattr(config, 'model_path', self.model_path)
                model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
                model.get_model().initialize_uni_modules(model.config, for_eval=True)
                vision_tower = model.get_vision_tower()
                vision_tower.load_model()
                vision_tower_aux = model.get_vision_tower_aux()
                vision_tower_aux.load_model()
                self.vl_model = model
                if not self.with_llm:
                    del model.lm_head
                    del model.model.embed_tokens
                    del model.model.layers
                    del model.model.norm

        from accelerate.utils import get_balanced_memory, infer_auto_device_map
        max_memory = get_balanced_memory(model,
                                         max_memory=self.max_memory,
                                         dtype=torch.half,
                                         no_split_module_classes=['CLIPEncoderLayer', 'ConvNeXtStage'])
        device_map = infer_auto_device_map(model,
                                           no_split_module_classes=['CLIPEncoderLayer', 'ConvNeXtStage'],
                                           max_memory=max_memory,
                                           dtype=torch.half)
        keys = ['model.vlm_uni_query_projector', 'model.vlm_uni_aux_projector', 'model.vlm_uni_val_projector']
        if keys[0] in device_map:
            for key in keys[1:]:
                device_map[key] = device_map[keys[0]]

        from accelerate import load_checkpoint_and_dispatch
        with disable_logging():
            load_checkpoint_and_dispatch(model=model,
                                         checkpoint=self.model_path,
                                         device_map=device_map if not self.with_llm else {'': 'cpu'},
                                         no_split_module_classes=['CLIPEncoderLayer', 'ConvNeXtStage'],
                                         dtype=torch.half)

        if keys[0] in device_map:
            add_device_hook(vision_tower, device_map[keys[0]])
            add_device_hook(vision_tower_aux, device_map[keys[0]])

        image_processor = model.model.vision_tower.image_processor
        if hasattr(model.config, 'image_size_aux'):
            if not hasattr(image_processor, 'image_size_raw'):
                image_processor.image_size_raw = image_processor.crop_size.copy(  # noqa
                )
            image_processor.crop_size['height'] = model.config.image_size_aux
            image_processor.crop_size['width'] = model.config.image_size_aux
            image_processor.size['shortest_edge'] = model.config.image_size_aux

        self.model = model.eval()
        self.image_processor = image_processor
        self.process_images = process_images

    def preprocess(self, messages: List[Dict]) -> List[Dict]:
        return messages

    @torch.no_grad()
    def forward(self, messages: List[Dict], max_batch_size: int = 1) -> List[Dict]:
        """extract image feature. ONLY implement it when the backend is
        turbomind engine.

        Args:
            messages(List[Dict]): the outputs of `preprocess`
            max_batch_size(int): the max batch size when forwarding vision
                model
        Return:
            the message list with forwarding results included
        """
        images = []
        for message in messages:
            if not isinstance(message['content'], List):
                continue
            _ = [x['image'] for x in message['content'] if x['type'] == 'image']
            assert len(_) == 1, f'MiniGeminiLlama accepts ONE input ' \
                f'image, but got {len(images)} images'
            images.extend(_)

        image_tensor = self.process_images(images, self.image_processor, self.model.config)
        image_grid = getattr(self.model.config, 'image_grid', 1)
        if hasattr(self.model.config, 'image_size_aux'):
            raw_shape = [
                self.image_processor.image_size_raw['height'] * image_grid,
                self.image_processor.image_size_raw['width'] * image_grid
            ]
            image_tensor_aux = image_tensor
            image_tensor = torch.nn.functional.interpolate(image_tensor,
                                                           size=raw_shape,
                                                           mode='bilinear',
                                                           align_corners=False)
        else:
            image_tensor_aux = []

        if image_grid >= 2:
            raw_image = image_tensor.reshape(3, image_grid, self.image_processor.image_size_raw['height'], image_grid,
                                             self.image_processor.image_size_raw['width'])
            raw_image = raw_image.permute(1, 3, 0, 2, 4)
            raw_image = raw_image.reshape(-1, 3, self.image_processor.image_size_raw['height'],
                                          self.image_processor.image_size_raw['width'])

            if getattr(self.model.config, 'image_global', False):
                global_image = image_tensor
                if len(global_image.shape) == 3:
                    global_image = global_image[None]
                global_image = torch.nn.functional.interpolate(
                    global_image,
                    size=[self.image_processor.image_size_raw['height'], self.image_processor.image_size_raw['width']],
                    mode='bilinear',
                    align_corners=False)
                # [image_crops, image_global]
                raw_image = torch.cat([raw_image, global_image], dim=0)
            image_tensor = raw_image.contiguous()
            image_tensor = image_tensor.unsqueeze(0)

        if type(image_tensor) is list:
            image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
            image_tensor_aux = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor_aux]
            logger.info(f'vision forward bs: {len(image_tensor)}')
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
            image_tensor_aux = image_tensor_aux.to(self.model.device, dtype=torch.float16)
            logger.info(f'vision forward shape: {image_tensor.shape}')
        images_embeds = self.model.encode_images(image_tensor, image_tensor_aux)

        outputs = torch.split(images_embeds, 1, dim=0)
        outputs = [x.squeeze() for x in outputs]
        messages.append(dict(role='forward', cotent=outputs))

    @staticmethod
    def proc_messages(messages, chat_template, sequence_start):
        """apply chat template to get the prompt."""
        prompt_messages = []
        IMAGE_TOKEN = '<IMAGE_TOKEN>'
        for message in messages:
            if isinstance(message['content'], str):
                prompt_messages.append(message)
                continue
            elif message['role'] in ['images', 'preprocess', 'forward']:
                continue
            n_images = len([1 for x in message['content'] if x['type'] == 'image'])
            content = [item['text'] for item in message['content'] if item['type'] == 'text']
            prompt = (IMAGE_TOKEN + '\n') * n_images + content[0]
            prompt_messages.append(dict(role='user', content=prompt))
        prompt = chat_template.messages2prompt(prompt_messages, sequence_start)
        return prompt, IMAGE_TOKEN

    def to_pytorch(self, messages, chat_template, tokenizer, sequence_start):
        assert 0, 'cogvlm is not supported by pytorch engine'

    def to_turbomind(self, messages, chat_template, tokenizer, sequence_start):
        prompt, IMAGE_TOKEN = self.proc_messages(messages, chat_template, sequence_start)
        return self.to_turbomind_aux(messages, prompt, IMAGE_TOKEN, tokenizer, sequence_start)
