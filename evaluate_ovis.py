import os
import json
import torch
from tqdm import tqdm
from PIL import Image
from moviepy.editor import VideoFileClip
from transformers import AutoModelForCausalLM

# --- Constants ---
model_name = 'AIDC-AI/Ovis2-8B'
max_num_frames = 12
max_partition = 1

# --- Load model and tokenizers ---
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    multimodal_max_length=32768,
    trust_remote_code=True
).cuda()
text_tokenizer = model.get_text_tokenizer()
visual_tokenizer = model.get_visual_tokenizer()

def sample_video_frames(video_path, num_frames=max_num_frames):
    with VideoFileClip(video_path) as clip:
        total_frames = int(clip.fps * clip.duration)
        if total_frames <= num_frames:
            indices = list(range(total_frames))
        else:
            stride = total_frames / num_frames
            indices = [min(total_frames - 1, int((stride * i + stride * (i + 1)) / 2)) for i in range(num_frames)]
        frames = [clip.get_frame(i / clip.fps) for i in indices]
        return [Image.fromarray(f, mode='RGB') for f in frames]

def run_ovis_mcq(root_path, input_json_file, output_file, mcq_type="single"):
    print(f"Evaluating videos in root path: {root_path} with type: {mcq_type}")
    # ---------- 1. Load data and build the global action list ----------
    with open(input_json_file, "r", encoding="utf-8") as f:
        video_data = json.load(f)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results = []

    for entry in tqdm(video_data, desc=f"Evaluating {input_json_file}", total=len(video_data)):
        video_name = entry["video_name"]
        video_path = os.path.join(root_path, video_name)
        choices = entry["choices"]
        choices_block = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])

        # Compose the MCQ prompt
        if mcq_type == "single":
            question = (
                "The video shows an industrial or surveillance scene. One or more people or vehicles may be present. From the actions listed below, select the one main action most clearly performed in this video. Review the video and captions. Only one action is correct. Reply with the action number only.\n"
                + choices_block
            )
        else:
            question = (
                "The video shows an industrial or surveillance scene. One or more people or vehicles may be present. From the actions listed below, select all actions that are being performed in this video (multiple may be correct). Review the video and captions. Reply with the action numbers only, separated by commas.\n"
                + choices_block
            )

        try:
            images = sample_video_frames(video_path, num_frames=max_num_frames)
            query = "\n".join(["<image>"] * len(images)) + "\n" + question

            prompt, input_ids, pixel_values = model.preprocess_inputs(
                query, images, max_partition=max_partition
            )
            attention_mask = (input_ids != text_tokenizer.pad_token_id)
            input_ids = input_ids.unsqueeze(0).to(model.device)
            attention_mask = attention_mask.unsqueeze(0).to(model.device)
            if pixel_values is not None:
                pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    max_new_tokens=1024,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=text_tokenizer.pad_token_id,
                    eos_token_id=model.generation_config.eos_token_id
                )[0]
                response = text_tokenizer.decode(output_ids, skip_special_tokens=True)
        except Exception as e:
            print(f"Ovis2 failed on {video_name}: {e}")
            continue

        correct_label = entry["answer_index"] + 1 if mcq_type == "single" else [i + 1 for i in entry["answer_indices"]]
        chosen_gt_actions = entry["chosen_gt_action"] if mcq_type == "single" else entry["chosen_gt_actions"]
        other_gt_actions = entry.get("other_gt_actions", [])

        results.append({
            "video_name": video_name,
            "question": question,
            "choices": choices_block,
            "model_response": response,
            "correct_label(s)": correct_label,
            "chosen_gt_action(s)": chosen_gt_actions,
            "other_gt_actions": other_gt_actions
        })

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

# Example calls – set your paths and files
run_ovis_mcq('/path_to_isafety_videos/normal', "/path_to_mcq/normal_mcq_multi.json", "isafety_results_ovis/normal_multi_results.json", mcq_type="multi")
run_ovis_mcq('/path_to_isafety_videos/normal', "/path_to_mcq/normal_mcq_single.json", "isafety_results_ovis/normal_single_results.json", mcq_type="single")
run_ovis_mcq('/path_to_isafety_videos/hazard', "/path_to_mcq/hazard_mcq_multi.json", "isafety_results_ovis/hazard_multi_results.json", mcq_type="multi")
run_ovis_mcq('/path_to_isafety_videos/hazard', "/path_to_mcq/hazard_mcq_single.json", "isafety_results_ovis/hazard_single_results.json", mcq_type="single")

