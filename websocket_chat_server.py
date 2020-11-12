# WS server chat

import asyncio
import websockets

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead

model_name = "sberbank-ai/rugpt2large" # "sberbank-ai/rugpt3large_based_on_gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelWithLMHead.from_pretrained(model_name)
model.to("cuda")

# string_init = '''- Привет. - сказал он.\n- Привет. - сказал я.\n'''
string_init = ''
string = string_init

async def chat(websocket, path):
    global string

    while True:
        msg = await websocket.recv()
        if msg == 'stop':
            await websocket.send('stopped!')
            asyncio.get_event_loop().stop()
            print(string)
            return
        elif msg == 'clear':
            await websocket.send('cleared!')
            string = string_init
            return
        elif msg == 'history':
            await websocket.send(f'history: \n{string}')
            return

        print(f'They: {msg}')
        string += f'- {msg} - сказал он.\n-'

        encoded_prompt = tokenizer.encode(string, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to("cuda")
        output_sequences = model.generate(
            input_ids=encoded_prompt,
            max_length=len(encoded_prompt[0]) + 50,
            temperature=1.,
            top_k=10,
            top_p=0.95,
            repetition_penalty=1.,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=50256,
        )

        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequence = output_sequences[0].tolist()
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        answer = text[len(string):].split('\n')[0]
        stripped_answer = ''.join([sec for inx, sec in enumerate(answer.split(' - ')) if not inx % 2]).strip()

        string += f'{answer}\n'

        print(f'Me: {stripped_answer}')
        await websocket.send(stripped_answer)

start_server = websockets.serve(chat, "192.168.1.3", 8765, ping_timeout=None, close_timeout=None)

asyncio.get_event_loop().run_until_complete(start_server)

print('waiting for message...')

asyncio.get_event_loop().run_forever()
