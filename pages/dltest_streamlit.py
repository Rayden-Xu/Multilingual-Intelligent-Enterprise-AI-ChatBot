import sys
sys.path.append("./SadTalker")
import os
import time
from time import strftime
import openai
import subprocess
import elevenlabs
import numpy as np
from elevenlabs import clone, generate, play, set_api_key
from elevenlabs import voices
import shutil
from SadTalker.src.utils.preprocess import CropAndExtract
from SadTalker.src.test_audio2coeff import Audio2Coeff  
from SadTalker.src.facerender.animate import AnimateFromCoeff
from SadTalker.src.facerender.pirender_animate import AnimateFromCoeff_PIRender
from SadTalker.src.generate_batch import get_data
from SadTalker.src.generate_facerender_batch import get_facerender_data
from SadTalker.src.utils.init_path import init_path
from argparse import Namespace
import streamlit as st
# API Keys

set_api_key(st.secrets['tts_key'])
openai.api_key= st.secrets['openai.api_key']
user_profile_url = "/mount/src/demo/src/biden_profile.png"
biden_profile_url = "/mount/src/demo/src/user_profile.png"

st.set_page_config(page_title="Chatbot", page_icon="💬")
st.header('Talk with President Joe Biden')

# Global variables
current_root_path = os.getcwd()
checkpoint_dir = "./checkpoints"
size = 256
old_version = "old_version_value"
preprocess = 'full'
device = "cuda"

#sadtalker initialize load model

@st.cache_data
def load_all_static_info():
    sadtalker_paths = init_path(checkpoint_dir, os.path.join(current_root_path, 'src/config'), size, old_version, preprocess)
    preprocess_model = CropAndExtract(sadtalker_paths, device)
    audio_to_coeff = Audio2Coeff(sadtalker_paths, device)
    animate_from_coeff = AnimateFromCoeff_PIRender(sadtalker_paths, device)
    all_voices = voices()
    ftvoice = next((v for v in all_voices if v.name == "Biden"), None)
    source_image_path = "./00117-2958808156.png" 
    result_dir ='./txt_result/video'
    save_dir = os.path.join(result_dir, strftime("%Y_%m_%d_%H.%M.%S"))

    return preprocess_model, audio_to_coeff, animate_from_coeff, ftvoice, save_dir,source_image_path


preprocess_model, audio_to_coeff, animate_from_coeff, ftvoice, save_dir,source_image_path = load_all_static_info()

def __init__():
    all_voices = voices()
    voice = next((v for v in all_voices if v.name == "Biden"), None)
    if not voice:
        raise ValueError("Voice 'Biden' not found in ElevenLabs voices.")

    print("Audio Processor initial done")
    

def convert_text_to_audio(text, output_audio_path):
    audio = generate(text=text, voice=ftvoice)
    with open(output_audio_path, 'wb') as f:
        f.write(audio)


def run_sadtalker(source_image_path, driven_audio_path,preprocess_model, audio_to_coeff, animate_from_coeff):

    # Define default parameters and configurations for the model
    args = Namespace(
        ref_eyeblink=None,
        ref_pose=None,
        checkpoint_dir='./checkpoints',
        result_dir='./txt_result/video',
        pose_style=0,
        batch_size=50,
        size=256,
        expression_scale=1.0,
        input_yaw=None,
        input_pitch=None,
        input_roll=None,
        enhancer=None,
        background_enhancer=None,
        cpu=False,
        face3dvis=False,
        still=True,
        preprocess='full',
        verbose=False,
        old_version=False,
        facerender='pirender',
        net_recon='resnet50',
        init_path=None,
        use_last_fc=False,
        bfm_folder='./checkpoints/BFM_Fitting/',
        bfm_model='BFM_model_front.mat',
        focal=1015.0,
        center=112.0,
        camera_d=10.0,
        z_near=5.0,
        z_far=15.0
    )

    # Parse the arguments
    pose_style = args.pose_style
    batch_size = args.batch_size
    input_yaw_list = args.input_yaw
    input_pitch_list = args.input_pitch
    input_roll_list = args.input_roll
    ref_eyeblink = args.ref_eyeblink
    ref_pose = args.ref_pose

    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    # Extract 3D Morphable Model (3DMM) coefficients for the source image
    first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(source_image_path, first_frame_dir, preprocess,\
                                                                            source_image_flag=True, pic_size=size)
    
    # Create directory to save data related to the first frame
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    ref_eyeblink_coeff_path=None

    ref_pose_coeff_path=None

    
    #audio2ceoff
    batch = get_data(first_coeff_path, driven_audio_path, device, ref_eyeblink_coeff_path, still=args.still)
    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

    #### 5.968579053878784s
    
    # 3dface render
    if args.face3dvis:
        from src.face3d.visualize import gen_composed_video
        gen_composed_video(args, device, first_coeff_path, coeff_path, driven_audio_path, os.path.join(save_dir, '3dface.mp4'))

    ####6.203108310699463s
    
    #coeff2video
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, driven_audio_path, 
                                batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                expression_scale=args.expression_scale, still_mode=args.still, preprocess=args.preprocess, size=args.size, facemodel=args.facerender)
    
    video_path = save_dir+'_converted.mp4'
    if os.path.exists(video_path):
        os.remove(video_path)

    result = animate_from_coeff.generate(data, save_dir, source_image_path, crop_info, \
                                enhancer=args.enhancer, background_enhancer=args.background_enhancer, preprocess=args.preprocess, img_size=args.size)
    
    shutil.move(result, save_dir+'.mp4')
    print('The generated video is named:', save_dir+'.mp4')
    ffmpeg_cmd = f"ffmpeg -i {save_dir}.mp4 -c:v libx264 {save_dir}_converted.mp4"
    subprocess.call(ffmpeg_cmd, shell=True)
    st.video(save_dir+'_converted.mp4')
    #### 8.638267517089844


    if not args.verbose:
        shutil.rmtree(save_dir)


def run(answer, preprocess_model, audio_to_coeff, animate_from_coeff):
    # Ensure there's a directory for storing audio results

    reply_audio_path = "./tts_output/biden/output_audio.wav"
    convert_text_to_audio(answer, reply_audio_path)
    time_tts = time.time()
    print(time_tts-time_send)   
    animation_output = run_sadtalker(source_image_path, reply_audio_path, preprocess_model, audio_to_coeff, animate_from_coeff)
    time_fin = time.time()
    print(time_fin-time_send)
    print(f"Generated animation for reply at: {animation_output}")
    #st.video('/home/lulei/whisper/txt_result/video/2023_09_18_00.43.20_converted.mp4')
    # Display the chat history
    for i, v in enumerate(st.session_state['chat_history']):
        if i % 2 == 1:
            col1, col2 = st.columns([1, 10])
            with col1:
                st.image(biden_profile_url, width=50)
            with col2:
                st.write(f"{v}")
        else:
            col1, col2 = st.columns([1, 10])
            with col1:
                st.image(user_profile_url, width=50)
            with col2:
                st.write(f"{v}")    
    


def continue_conversation(conversation_history, user_message):
    conversation_history.append({"role": "user", "content": user_message})
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=conversation_history
    )
    assistant_message = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": assistant_message})
    return assistant_message

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = [
        ''' My fellow Americans,and supporters: I'm Joe Biden, I extend to you my deepest respect and gratitude. It is through the participation and commitment of each and every one of you that our democracy continues to thrive. '''
    ]

if 'conversation_history' not in st.session_state:
    with open('biden_setting_candid.txt', 'r') as b_file:
        role1 = b_file.read()
    st.session_state['conversation_history'] = [
        {"role": "system", "content": role1}
    ]


def limit_words(text, limit=20):
    words = text.split()
    if len(words) > limit:
        return " ".join(words[:limit]) + "..."
    return text


if __name__ == '__main__':
    user_message = st.text_input("Ask President Joe Biden a question:")
    if st.button('Send'):
        time_send = time.time()
        st.session_state['chat_history'].append(user_message)
        answer = continue_conversation(st.session_state['conversation_history'], user_message)
        truncated_answer = limit_words(answer)
        st.session_state['chat_history'].append(answer)
        time_chat = time.time()
        print(time_chat-time_send)
        run(answer, preprocess_model, audio_to_coeff, animate_from_coeff)

