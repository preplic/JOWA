import sys

import numpy as np
import torch
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


@torch.no_grad()
def make_reconstructions_from_batch(
    batch, 
    save_dir, 
    epoch, 
    tokenizer, 
):
    original_frames = tensor_to_np_frames(
        rearrange(batch['observations'], 'b t c h w -> b t h w c')
    )
    rec_frames = generate_reconstructions_with_tokenizer(
        batch, 
        tokenizer, 
    )

    res = np.concatenate((original_frames, rec_frames), axis=-2)
    res = rearrange(res, 'b t h w c -> (b t) h w c')
    res = np.squeeze(res, axis=-1)  # due to gray scale
    for i, image in enumerate(res):
        img = Image.fromarray(image)
        img.save(save_dir / f'epoch_{epoch:03d}_t_{i:03d}.png')


def insert_separators(imgs, separator_width=1, separator_color=255):
    b, t, h, w, c = imgs.shape
    separator = np.full((h, separator_width, c), separator_color, dtype=imgs.dtype)
    imgs_with_separators = []
    for batch in imgs:
        batch_with_separators = [batch[0]]
        for img in batch[1:]:
            batch_with_separators.append(separator)
            batch_with_separators.append(img)
        imgs_with_separators.append(np.concatenate(batch_with_separators, axis=1))
    return np.stack(imgs_with_separators)


def add_text_label(img_array, label, label_width=120, font_size=10):
    """Adds a text label to the left of a batch of images using PIL."""
    b, h, w, c = img_array.shape
    # Create a white canvas for the label
    label_canvas_np = np.full((b, h, label_width, c), 255, dtype=img_array.dtype)

    # Squeeze single channel if it exists for PIL compatibility
    if c == 1:
        label_canvas_np = label_canvas_np.squeeze(-1)

    # Try to load a good font, fallback to a default one
    try:
        # A common font in Linux environments
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except IOError:
        # Fallback for other environments or if font is not found
        font = ImageFont.load_default()

    # Add text to each image in the batch
    for i in range(b):
        # Convert numpy canvas to PIL Image
        pil_img = Image.fromarray(label_canvas_np[i])
        draw = ImageDraw.Draw(pil_img)

        # Calculate text size to center it
        if hasattr(draw, 'textbbox'):
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            text_w, text_h = draw.textsize(label, font=font)

        text_x = (label_width - text_w) // 2
        text_y = (h - text_h) // 2
        draw.text((text_x, text_y), label, font=font, fill=0)  # Black text
        label_canvas_np[i] = np.array(pil_img)

    if c == 1:
        label_canvas_np = np.expand_dims(label_canvas_np, axis=-1)

    return np.concatenate((label_canvas_np, img_array), axis=2)

@torch.no_grad()
def make_reconstructions_of_trajectories(
    batch, 
    save_dir, 
    epoch, 
    tokenizer, 
    jowa, 
    separator_width=2,
):
    b, t = batch['observations'].shape[:2]
    original_frames = tensor_to_np_frames(
        rearrange(batch['observations'], 'b t c h w -> b t h w c')
    )
    original_concate_frames = insert_separators(
        original_frames, 
        separator_width=separator_width, 
        separator_color=255,
    )
    original_concate_frames = add_text_label(original_concate_frames, "Original")
    tokenizer_rec_frames = generate_reconstructions_with_tokenizer(
        batch, 
        tokenizer, 
    )
    tokenizer_concate_rec_frames = insert_separators(
        tokenizer_rec_frames, 
        separator_width=separator_width, 
        separator_color=255,
    )
    tokenizer_concate_rec_frames = add_text_label(tokenizer_concate_rec_frames, "Tokenizer")
    
    # teacher-forcing regression
    temp = tokenizer.encode(batch['observations'], should_preprocess=True)
    h, w = temp.z_quantized.shape[-2:]
    obs_tokens = temp.tokens  # (B, L, K)
    act_tokens = rearrange(batch['actions'], 'b l -> b l 1')
    tokens = rearrange(
        torch.cat((obs_tokens, act_tokens), dim=2), 
        'b l k1 -> b (l k1)',
    )  # (B, L(K+1))
    outputs = jowa(tokens, batch['envs'])
    
    def get_concate_rec_frames_from_outputs(outputs):
        logits_observations = outputs.logits_observations[:, :-1]
        tokens_observations = logits_observations.argmax(dim=-1)
        tokens_observations = torch.cat(
            (obs_tokens[:, 0, 0].unsqueeze(1), tokens_observations), 
            dim=1,
        )  # (B, t*h*w)
        embeddings_observations = tokenizer.embedding(tokens_observations.flatten())  # (B*t*h*w, E)
        e = embeddings_observations.size(1)
        z_q = rearrange(
            embeddings_observations, 
            '(b t h w) e -> (b t) e h w', 
            b=b, t=t, e=e, h=h, w=w,
        ).contiguous()
        output_frames = torch.clamp(
            tokenizer.decode(z_q, should_postprocess=True), 0, 1,
        )
        output_frames = rearrange(
            output_frames, 
            '(b t) c h w -> b t h w c', 
            b=b, t=t,
        )
        rec_frames = tensor_to_np_frames(output_frames)
        
        return insert_separators(
            rec_frames, 
            separator_width=separator_width, 
            separator_color=255,
        )
    
    teacher_forcing_regression_concate_rec_frames = get_concate_rec_frames_from_outputs(
        outputs
    )
    teacher_forcing_regression_concate_rec_frames = add_text_label(
        teacher_forcing_regression_concate_rec_frames, "Teacher-Forcing"
    )
    
    # auto-regression
    num_given_blocks = 4
    given_blocks_tokens = tokens[:, :num_given_blocks*(h * w + 1)]
    for step in tqdm(
        range(num_given_blocks * (h * w + 1), t * (h * w + 1)), 
        disable=True, 
        desc='auto-regression', 
        file=sys.stdout,
    ):
        if (step+1) % (h*w+1) == 0:
            given_blocks_tokens = torch.cat(
                (given_blocks_tokens, act_tokens[:, step // (h*w+1)]), 
                dim=1,
            )
        else:
            outputs = jowa(given_blocks_tokens, batch['envs'])
            logits_observations = outputs.logits_observations[:, -1]
            tokens_observations = logits_observations.argmax(dim=-1)
            given_blocks_tokens = torch.cat(
                (given_blocks_tokens, tokens_observations.unsqueeze(1)), 
                dim=1,
            )

    outputs = jowa(given_blocks_tokens, batch['envs'])
    auto_regression_concate_rec_frames = get_concate_rec_frames_from_outputs(outputs)
    auto_regression_concate_rec_frames = add_text_label(
        auto_regression_concate_rec_frames, "Auto-Regression"
    )
    
    # save
    separator = np.full(
        (b, separator_width, *original_concate_frames.shape[2:]), 
        255, 
        dtype=original_concate_frames.dtype,
    )
    res = np.concatenate(
        (
            original_concate_frames, separator, 
            tokenizer_concate_rec_frames, separator, 
            teacher_forcing_regression_concate_rec_frames, separator,
            auto_regression_concate_rec_frames,
        ), 
        axis=1,
    )
    res = np.squeeze(res, axis=-1)  # due to gray scale
    
    # 保存3张图片: 第一张、中间一张和最后一张
    indices_to_save = sorted(list(set([0, len(res) // 2, len(res) - 1])))
    for i in indices_to_save:
        image = res[i]
        img = Image.fromarray(image)
        img.save(save_dir / f'{str(jowa)}_epoch_{epoch:03d}_t_{i:03d}.png')


def tensor_to_np_frames(inputs):
    check_float_btw_0_1(inputs)
    return inputs.to(float).mul(255).cpu().numpy().astype(np.uint8)


def check_float_btw_0_1(inputs):
    assert inputs.is_floating_point() and (inputs >= 0).all() and (inputs <= 1).all()


@torch.no_grad()
def generate_reconstructions_with_tokenizer(batch, tokenizer):
    inputs = rearrange(batch['observations'], 'b t c h w -> (b t) c h w')
    outputs = reconstruct_through_tokenizer(inputs, tokenizer)
    b, t, _, _, _ = batch['observations'].size()
    outputs = rearrange(outputs, '(b t) c h w -> b t h w c', b=b, t=t)
    rec_frames = tensor_to_np_frames(outputs)
    return rec_frames


@torch.no_grad()
def reconstruct_through_tokenizer(inputs, tokenizer):
    check_float_btw_0_1(inputs)
    reconstructions = tokenizer.encode_decode(
        inputs, 
        should_preprocess=True, 
        should_postprocess=True,
    )
    return torch.clamp(reconstructions, 0, 1)
