#!/usr/bin/env python3

import os
import sys
sys.path += [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')]
sys.path += [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))]

import fire
import json
import os
import numpy as np
import tensorflow as tf
import tflex
import argparse
import re
import csv
import shutil
import time
from datetime import datetime

import model, sample, encoder

def interact_model(
    model_name='117M',
    restore_from=None,
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=0.0,
    penalize=0,
    prefix=None,
    include_prefix=True,
    split_context=0.5,
    sample_dir='samples',
    return_as_list=False,
    truncate=None,
    destination_path='gpt_2_gen_texts.txt',
    sample_delim='=' * 20 + '\n'
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :top_p=0.0 : Float value controlling diversity. Implements nucleus sampling,
     overriding top_k if set to a value > 0. A good setting is 0.9.
    :penalize=0.0 : Float value controlling "used" penalty. Implements repetition
     reduction (similar to CTRL) if set to a value > 0. A decent setting might be 0.85
     with temperature 0.3 and top_k 40.
    """
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    if nsamples == 1:
        sample_delim = ''

    if prefix == '':
        prefix = None
    
    

    if not length:
        assert truncate is not None, "If generating a non-fixed length \
                sample, must have a truncation term."
    assert 0 < split_context < 1

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))


# CREATE TF SESS

    with tflex.Session(graph=tf.Graph()) as sess:
        context = tf.compat.v1.placeholder(tf.int32, [batch_size, None])
        np.random.seed(None)
        tf.compat.v1.set_random_seed(seed)
        model.model(hparams=hparams, X=context)

        saver = tflex.Saver()
        if restore_from is None:
          restore_from = os.path.join('models', model_name)
        ckpt = tflex.latest_checkpoint(restore_from)
        saver.restore(sess, ckpt)

        context = tf.placeholder(tf.int32, [batch_size, None])

# ACTUAL GENERATION

        if prefix:
            prefix_enc = enc.encode(prefix)

        np.random.seed(None)
        tf.set_random_seed(seed)

        output = sample.sample_sequence(
            hparams=hparams,
            length=min((length), 1023 - (len(prefix_enc) if prefix else 0)),
            start_token=enc.encoder['<|endoftext|>'] if not prefix else None,
            context=context if prefix else None,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )[:, 1:]

        split_length = int(1023 * split_context)
        split_output_length = min(length, 1023 - split_length)
        split_output = sample.sample_sequence(
            hparams=hparams,
            length=split_output_length,
            start_token=enc.encoder['<|endoftext|>'] if not prefix else None,
            context=context if prefix else None,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )[:, 1:]

        if destination_path:
            f = open(destination_path, 'w')
        generated = 0
        gen_texts = []
        while generated < nsamples:
            gen_text = [np.array([])] * batch_size
            truncated = [False] * batch_size
            if prefix:
                context_tokens = [prefix_enc] * batch_size
            else: 
                context_tokens = [[enc.encoder['<|endoftext|>']]] * batch_size
            total_tokens = len(context_tokens[0])
            generated_once = False
            while False in truncated:
                num_tokens = 1023 - (len(context_tokens[0]))
                if generated_once:
                    new_split_output_length = min(length - total_tokens, 1023 - split_length)
                    if new_split_output_length != split_output_length: 
                        split_output = sample.sample_sequence(
                            hparams=hparams,
                            length=new_split_output_length,
                            start_token=enc.encoder['<|endoftext|>'] if not prefix else None,
                            context=context if prefix else None,
                            batch_size=batch_size,
                            temperature=temperature, top_k=top_k, top_p=top_p
                        )[:, 1:]
                    out = sess.run(split_output, feed_dict={
                        context: context_tokens
                        # TODO if a particular item in a batch is finished, its context can be dropped
                        # (but we need to keep track of which output index corresponds to which batch item then)
                    })

                else:
                    out = sess.run(output, feed_dict={
                        context: context_tokens
                    })
                total_tokens += num_tokens
                for i in range(batch_size):
                    text = out[i]
                    trunc_text = ""
                    if prefix: 
                        text = np.append(context_tokens[i][:1], text)
                    if truncate or all(gen_text):
                        context_tokens[i] = out[i][(1023 - split_length - 1):]
                        if generated_once:
                            text = out[i][split_length:]
                        if truncate:
                            to_trunc = enc.decode(text)
                            truncate_esc = re.escape(truncate)
                            if prefix and not include_prefix:
                                prefix_esc = re.escape(prefix)
                                pattern = '(?:{})(.*?)(?:{})'.format(prefix_esc,
                                                                    truncate_esc)
                            else:
                                pattern = '(.*?)(?:{})'.format(truncate_esc)
                            trunc_text = re.search(pattern, to_trunc, re.S)
                            if trunc_text:
                                text = enc.encode(trunc_text.group(1))
                                # better to re-encode here then decode every generation cycle, I think
                    if not truncated[i]:
                        gen_text[i] = np.concatenate((gen_text[i], text), axis=None)
                        if trunc_text or (length is not 0 and total_tokens >= length-1):
                            truncated[i] = True
                            gen = enc.decode(gen_text[i]).lstrip('\n')
                            if destination_path:
                                f.write("{}\n{}".format(gen, sample_delim))
                            if not return_as_list and not destination_path:
                                print("{}\n{}".format(gen, sample_delim), end='')
                            gen_texts.append(gen)
                generated_once = True
            generated += batch_size
        if destination_path:
            f.close()
        if return_as_list:
            return gen_texts



        """
        while True:
            
            if prompt is not None:
              if os.path.isfile(prompt):
                  with open(prompt) as f:
                      raw_text = f.read()
              else:
                  raw_text = prompt
            else:
                raw_text = input("Model prompt >>> ")
                if not raw_text:
                    raw_text="\n"
            if len(raw_text) > 1 and raw_text.endswith('\n'):
                raw_text = raw_text[:-1]
            print('Prompt:', repr(raw_text))
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    sys.stdout.write(raw_text)
                    print(text)
                    sys.stdout.flush()
            print("=" * 80)
"""


if __name__ == '__main__':
    fire.Fire(interact_model)


"""
def cmd():
    #Function called when invoking from the terminal.
    parser = argparse.ArgumentParser(
        description="Easily retrain OpenAI's GPT-2 text-generating model on new texts. (https://github.com/minimaxir/gpt-2-simple)"
    )
    parser.add_argument(
        '--restore_from',  help="[finetune] Whether to load model 'fresh' or from 'latest' checkpoint.",
        nargs='?', default='latest')
    parser.add_argument(
        '--model_name',  help="[finetune] Name of the GPT-2 model to finetune",
        nargs='?', default='124M')
    parser.add_argument(
        '--nsamples',  help="[generate] How many texts to generate.",
        nargs='?', default=1, type=int)
    parser.add_argument(
        '--batch_size',  help="[generate] Batch size for generation (increase for GPUs)",
        nargs='?', default=1, type=int)
    parser.add_argument(
        '--length',  help="[generate] Length (tokens) of the generated texts",
        nargs='?', default=1023, type=int)
    parser.add_argument(
        '--temperature',  help="[generate] Temperature of the generated texts",
        nargs='?', default=0.7, type=float)
    parser.add_argument(
        '--top_k',  help="[generate] Sample only from top k tokens",
        nargs='?', default=0, type=int)
    parser.add_argument(
        '--top_p',  help="[generate] Sample from top p prob (overrides top_k if nonzero)",
        nargs='?', default=0.0, type=float)
    parser.add_argument(
        '--prefix',  help="[generate] Prefix for generated texts",
        nargs='?', default=None)
    parser.add_argument(
    '--prompt',  help="[generate] Prefix for generated texts",
    nargs='?', default=None)
"""
