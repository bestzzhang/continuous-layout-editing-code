import gradio as gr
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from utils.ptp_utils import AttentionStore
from utils.ddim_inversion import create_inversion_latents
from single_image_learning import run as run1
from run_sample import run as run2
from run_sample import load_model
import torch
import json
import os
import random

# -------------------------------------Helper Functions-----------------------------------------
def seg_image(img, color2token, bg_color):
    global seg_processor
    global seg_model
    colors = []
    texts = []
    for color, token in color2token:
        colors.append(np.array(list(color)))
        texts.append(token)

    texts.append("background")
    inputs = seg_processor(text=texts, images=[img] * len(texts), padding=True, return_tensors="pt")
    outputs = seg_model(**inputs)
    logits = np.argmax(outputs.logits.detach().numpy(), axis=0)
    img = np.zeros((352, 352, 3))
    img[:,:] = bg_color
    for i, token in enumerate(texts[:-1]):
        pred_i = logits == i
        img[pred_i] = colors[i]
    image = Image.fromarray(img.astype(np.uint8)).resize((64,64), resample = 0)
    image.save('UI/ini_mask.png')
    return image

def get_masks(color2token, mask_image, size = (16,16)):
    masks = []
    pil_image = Image.open(mask_image).resize(size,resample=0)
    img = np.array(pil_image)
    for k, _ in color2token:
        color = np.array(k)
        mask = np.all(img == color, axis=-1)
        assert mask.sum()!=0
        masks.append(mask)
    return masks

def get_blend_mask(tgt_bg):
    m1 = np.array(Image.open("UI/ini_mask.png"))
    m2 = np.array(Image.open(f"UI/target_layout/tgt_mask{layout_count}.png"))
    b = np.array(tgt_bg)
    blend_mask = np.all(m1 == b, axis=-1) & np.all(m2 == b, axis=-1)
    blend_mask = torch.from_numpy(blend_mask).cuda()
    return blend_mask

def prompt_to_idx(prompt):
    words = list(prompt.strip(" ").replace(", ", ",").replace(",", " , ").split(" "))
    # "cat,dog" -> ["cat", ",", "dog"]
    # "cat, dog" -> ["cat", ",", "dog"]
    if len(words) > 1:
        return str(dict(zip(range(1, 1+len(words)), words)))
    return ""

def RGB_to_Hex(rgb):
    if rgb[0]>=250 and rgb[1]>=250 and rgb[2]>=250:
        return "#FFFFFF"
    color = "#"
    for i in rgb:
        num = int(i)
        color += str(hex(num))[-2:].replace("x", "0").upper()
    return color

def get_hex_colors(mask):
    global colors
    colors = []
    hex_colors = []
    image_colors = mask.getcolors(maxcolors=64*64)
    for i, (_, color) in enumerate(image_colors):
        hex_color = RGB_to_Hex(color)
        if hex_color != "#FFFFFF":
            colors.append(color)
            hex_colors.append(hex_color)
        else:
            white = color
    colors.append(white)
    return hex_colors

def clean_image(image, num_color):
    image = image.resize((64,64), resample=0)
    X = np.array(image).reshape(-1,3)
    kmeans = KMeans(n_clusters=num_color, random_state=0, n_init='auto').fit(X)
    for i in range(num_color):
        t = kmeans.labels_ == i
        mean_color = X[t].mean(axis = 0).astype(int)
        X[t] = mean_color
    X = X.reshape(64,64,3)
    return Image.fromarray(X)

def generate(prompt, pipe, blend_mask, masks, indices_to_alter):
    inversion_latents = create_inversion_latents(pipe, "UI/initial.png", prompt, \
                            guidance_scale=5, ddim_steps=50)
    blend_dict = {
        "blend_mask":blend_mask,
        "inversion_latents":inversion_latents,
        "blend_steps":15
    }
    seed = random.randint(0, 9999)
    g = torch.Generator('cuda').manual_seed(seed)
    controller = AttentionStore()    
    image = run2(pipe,
                prompt=prompt,
                guidance_scale = 5,
                n_inference_steps = 50,
                eta = 0,
                controller=controller,
                indices_to_alter= indices_to_alter,
                generator=g,
                run_standard_sd=False,
                scale_factor = 20,
                thresholds = {0:0.6, 10: 0.7, 20: 0.8},
                max_iter_to_alter=25,
                max_refinement_steps=40,
                scale_range = (1., 0.5),
                masks = masks,
                blend_dict = blend_dict,
                )
    image.save(f"UI/images/{layout_count}.png")
    return image

def train_model(color2token, train_prompt, out_dir):
    from config import RunConfig

    args = RunConfig()
    args.image_path = "UI/initial.png"
    args.output_dir = out_dir
    os.makedirs(out_dir, exist_ok = True)

    args.iv_initializer_tokens = [token for _, token in color2token]
    args.iv_modifier_tokens = [f"<{token}>" for token in args.iv_initializer_tokens]
    masks = get_masks(color2token, "UI/ini_mask.png",(64,64))
    args.iv_mask = np.stack(masks)[:,None,:,:]

    args.ft_initializer_tokens = ["sks","uy"]
    args.ft_modifier_tokens = [f"<new{i}>" for i in range(len(args.ft_initializer_tokens))]
    tail_str = " ".join(args.ft_modifier_tokens)

    args.reg_dirs = []
    for i in args.iv_initializer_tokens:
        path = f"real_reg/samples_{i}"
        if os.path.exists(path):
            args.reg_dirs.append(path)

    train_prompts = []    
    all_replaced_prompt = train_prompt
    for initializer_token, modifier_token in zip(args.iv_initializer_tokens, args.iv_modifier_tokens):
        all_replaced_prompt = all_replaced_prompt.replace(initializer_token, modifier_token)
        train_prompts.append(train_prompt.replace(initializer_token, modifier_token))
    train_prompts.append(all_replaced_prompt + " " + tail_str)
    args.train_prompt = train_prompts
    print(args.train_prompt)

    run1(args)

def replace_color(path, old_mapping, new_mapping):
    image = np.array(Image.open(path))
    for k in old_mapping:
        v = old_mapping[k]
        m = np.all(image == v, axis=-1)
        image[m] = np.array(new_mapping[k])
    image = Image.fromarray(image)
    image.save(path)
    return image
# --------------------------------------------------------------------------------------------

def generateImage(ini_image, text, obj1, obj2, obj3, obj4):
    global colors
    global pipe
    global global_idxs_to_alter
    global new_photo
    global seg_mask
    global layout_count
    
    bg_color = colors[-1]
    # mapping from token to color
    mapping = dict(zip([obj1, obj2, obj3, obj4], colors))
    mapping["background"] = bg_color

    color2token = []
    # reorder the mapping in the order of the tokens appear in the sentence
    for choice in choices:
        color2token.append((mapping[choice], choice))

    if new_photo:
        layout_count = 0
        torch.cuda.empty_cache()
        new_photo = False
        out_dir = "UI/embeds/"
        # use clip segmentation for getting segmentatin mask
        seg_mask = seg_image(ini_image, color2token, bg_color) 
        # run learning code
        train_model(color2token, text, out_dir)
        # load trained model
        delta_ckpt = out_dir + "fine_tune/delta.bin"
        pipe, _ = load_model(delta_ckpt)
        print("model loaded")
    else:
        # replace the colors of mask with the new color
        with open("UI/semantic_dict.json", 'r') as f:
            old_mapping = json.load(f)
        for i in range(layout_count):
            replace_color(f"UI/target_layout/tgt_mask{i}.png", old_mapping, mapping)
        seg_mask = replace_color("UI/ini_mask.png", old_mapping, mapping)

    # save color2token to semantic_dict.json
    with open("UI/semantic_dict.json", 'w') as f:
        json.dump(mapping, f)

    tokens = [token for _, token in color2token]
    for token in tokens:
        text = text.replace(token, f"<{token}>")
    blend_mask = get_blend_mask(bg_color)
    masks = [torch.Tensor(mask).cuda() for mask in get_masks(color2token, f"UI/target_layout/tgt_mask{layout_count}.png")]
    
    indices_to_alter = [int(idx) for idx in global_idxs_to_alter]

    image = generate(text, pipe, blend_mask, masks, indices_to_alter) 
    
    outputs = [gr.update(value=image), gr.update(value=seg_mask),gr.update(value="UI/initial.png"), gr.update(value=f"UI/target_layout/tgt_mask{layout_count}.png")]
    labels = [gr.update(label=f"color1({obj1})"), gr.update(label=f"color2({obj2})"),gr.update(label=f"color3({obj3})"), gr.update(label=obj4)]
    layout_count += 1
    return outputs + labels


def upload_file(files):
    global new_photo
    file_paths = [file.name for file in files]
    file_path = file_paths[0]
    image = Image.open(file_path).resize((512,512))
    image.save("UI/initial.png")
    new_photo = True
    return gr.update(visible=True, value=image)

def fetch_colors(image, idxs_to_alter, prompt):
    global global_idxs_to_alter
    global choices
    global_idxs_to_alter = idxs_to_alter.split(',')
    tokens = prompt.strip(" ").replace(", ", ",").replace(",", " , ").split(" ")
    mask = clean_image(image, len(global_idxs_to_alter)+1)
    mask.save(f"UI/target_layout/tgt_mask{layout_count}.png")
    hex_colors = get_hex_colors(mask)

    choices = [tokens[int(idx)-1] for idx in global_idxs_to_alter]
    visibility = []
    values = []
    for i in range(4):
        if i < len(hex_colors):
            visibility.append(True)
            values.append(hex_colors[i])
        else:
            visibility.append(False)
            values.append(None)

    # update the Dropdown
    ret = [gr.Dropdown.update(choices=choices,visible=v1) for v1 in visibility]
    # update the color selector
    ret += [gr.update(visible=v1, value=v2) for v1,v2 in zip(visibility,values)]
    return ret

def clear_image():
    res = [gr.update(value = None) for _ in range(4)] + [gr.update(visible=False) for _ in range(8)]
    return res

colors = []
global_idxs_to_alter = []
choices = []
new_photo = False
seg_mask = None
layout_count = 0

seg_processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
seg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
print("segmentation model loaded")
os.makedirs("UI/target_layout/", exist_ok=True)
os.makedirs("UI/images/", exist_ok=True)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            with gr.Box():
                tgt_layout = gr.Image("canvas.png", source = "canvas", tool="color-sketch", type="pil", shape=(256, 256))
                upload_button = gr.UploadButton("Click to Upload a Image", file_types=["image"], file_count="multiple")
                prompt = gr.Textbox(label="Prompt")
                idxs_to_alter = gr.Textbox(label="Index of the objects")
                with gr.Row():
                    token1 = gr.Dropdown(visible=False,label="color1_object",interactive = True)
                    token2 = gr.Dropdown(visible=False,label="color2_object",interactive = True)
                    token3 = gr.Dropdown(visible=False,label="color3_object",interactive = True)
                    token4 = gr.Dropdown(visible=False,label="color4_object",interactive = True)
                with gr.Row():
                    fetch = gr.Button(value="Getcolor")
                    submit = gr.Button(value="Submit")
                    clear = gr.Button(value="Clear")

        with gr.Column():
            with gr.Box():
                tokenIndex = gr.Textbox(label="Token Index")
                with gr.Row():
                    input_image = gr.Image(label="Input Image", visible=True,interactive = False, shape=(256, 256))
                    input_mask = gr.Image(label="Input Mask", visible=True,interactive = False, shape=(256, 256))
                with gr.Row():
                    output_image = gr.Image(label="Output Image",visible=True,interactive = False, shape=(256, 256))
                    output_mask = gr.Image(label="Output Mask",visible=True,interactive = False, shape=(256, 256))
                with gr.Row():
                    c1 = gr.ColorPicker(visible=False,label="color1")
                    c2 = gr.ColorPicker(visible=False,label="color2")
                    c3 = gr.ColorPicker(visible=False,label="color3")
                    c4 = gr.ColorPicker(visible=False,label="color4")
    
    submit_inputs = [
        input_image,
        prompt,
        token1,
        token2,
        token3,
        token4,
    ]

    fetch_outputs = [
        token1,
        token2,
        token3,
        token4,
        c1,
        c2,
        c3,
        c4
    ]
    prompt.change(prompt_to_idx, prompt, tokenIndex)
    fetch.click(fetch_colors, 
                inputs=[tgt_layout,idxs_to_alter,prompt], 
                outputs=fetch_outputs)
    upload_button.upload(upload_file, upload_button, input_image)
    submit.click(generateImage, inputs=submit_inputs, outputs=[output_image,input_mask, input_image, output_mask,c1,c2,c3,c4])
    clear.click(clear_image, outputs=[tgt_layout, output_image, input_mask, output_mask] + fetch_outputs)

if __name__ == "__main__":
    demo.launch()
