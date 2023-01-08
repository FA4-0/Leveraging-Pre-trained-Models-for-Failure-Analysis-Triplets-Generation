import os
import numpy as np
from os.path import join
path = os.getcwd()
year = 2019
absoulte_dir = [join(path, f'plm/finetuning/{year}'), join(path, f'plm/use_weight/{year}')]
MODEL_CLASSES = {
                    'facebook/bart-large-cnn': 'bart',
                    'bert-base-uncased': 'bert', 
                    'roberta-base': 'roberta', 
                    'openai-gpt': 'gpt3',
                    'gpt2': 'gpt2-b',
                    'gpt2-medium': 'gpt2-m',
                    'gpt2-large': 'gpt2-l',
                    }

result_files = {}
for dir_ in absoulte_dir:
    for i, j in MODEL_CLASSES.items():
        if dir_.split('/')[-2].lower() == 'finetuning':
            result_files[f'{j}'] = join(dir_, f"{i.split('/')[0]}/evaluation")
        elif dir_.split('/')[-2].lower() == 'use_weight':
            result_files[f'w_{j}'] = join(dir_, f"{i.split('/')[0]}/evaluation")
        else:
            result_files[f"{dir_.split('/')[-2].lower()[:3]}_{j}"] = join(dir_, f"{i.split('/')[0]}/evaluation")

eval_metric = {}
for i, j in result_files.items():
    npys = [x for x in os.listdir(j) if '.npy' in x]
    eval_metric[i] = {}
    for npy in npys:
        if not 'lese' in npy:
            eval_metric[i][f"{npy.split('.npy')[0]}"] = np.load(join(j, npy), allow_pickle = True)
        else:
            eval_metric[i][f"{npy.split('.npy')[0]}"] = np.load(join(j, npy), allow_pickle = True).ravel()[0]
        if 'rouge' in npy:
            eval_metric[i][f"{npy.split('.npy')[0]}"] = np.load(join(j, npy), allow_pickle = True).ravel()[0]
    
print('+-------------+-------------+-------------+-------------+--------------------+--------------------+--------------------+-------------+--------------------+---------------+---------------+')
print('|  Model      |   BLEU-1    |    BLEU-3   |   MET.      |      ROUGE-1       |      ROUGE-L       |        LESE-1      |    Lev-1    |  LESE-3            |   Lev-3       |      PPL      |')
print('|            |              |             |             | Prec. | Rec. | F1  | Prec. | Rec. | F1  | Prec. | Rec. | F1  |             | Prec. | Rec. | F1  |               |               |')
print('+------------+-------------+-------------+-------------+-------+------+-----+--------+------+-----+-------+------+-----+-------------+-------+------+-----+---------------+---------------+')
for i, j in eval_metric.items():
    ppl = round(j['perplexity'][-1], 2)
    bleuscore_ = ''.join([x for x in j.keys() if 'bleuscore_' in x]) if bool([x for x in j.keys() if 'bleuscore_' in x]) else '-'
    bleuscore_ = round(np.mean(j[bleuscore_])*100, 2) if bleuscore_ != '-' else '-'
    bleuscore3_ = ''.join([x for x in j.keys() if 'bleuscore3_' in x]) if bool([x for x in j.keys() if 'bleuscore3_' in x]) else '-'
    bleuscore3_ = round(np.mean(j[bleuscore3_])*100, 2) if bleuscore3_ != '-' else '-'
    meteor_ = ''.join([x for x in j.keys() if 'meteor_' in x]) if bool([x for x in j.keys() if 'meteor_' in x]) else '-'
    meteor_ = round(np.mean(j[meteor_])*100, 2) if meteor_ != '-' else '-'
    rouge_ = ''.join([x for x in j.keys() if 'rouge_' in x]) if bool([x for x in j.keys() if 'rouge_' in x]) else '-'
    rouge_ = j[rouge_] if rouge_ != '-' else '-'
    rouge1_ = rouge_['rouge-1'] if rouge_ != '-' else '-'
    p1_, r1_, f1_1_ = round(rouge1_['p']*100, 2) if rouge1_ != '-' else '-', round(rouge1_['r']*100, 2) if rouge1_ != '-' else '-', \
                        round(rouge1_['f']*100, 2) if rouge1_ != '-' else '-'
    rougel_ = rouge_['rouge-l'] if rouge_ != '-' else '-'
    pl_, rl_, f1_l_ = round(rougel_['p']*100, 2) if rougel_ != '-' else '-', round(rougel_['r']*100, 2) if rougel_ != '-' else '-', \
                        round(rougel_['f']*100, 2) if rougel_ != '-' else '-'
    lese_ = ''.join([x for x in j.keys() if 'lese1_' in x]) if bool([x for x in j.keys() if 'lese1_' in x]) else '-'
    lese_ = j[lese_] if lese_ != '-' else '-'
    pls_, rls_, f1_ls_, levd1_ = round(np.mean(lese_['prec_lev'])*100, 2) if lese_ != '-' else '-', \
                                    round(np.mean(lese_['rec_lev'])*100, 2) if lese_ != '-' else '-', \
                                    round(np.mean(lese_['fs_lev'])*100, 2) if lese_ != '-' else '-', \
                                    np.mean(lese_['lev_d'])//1 if lese_ != '-' else '-'
    lese3_ = ''.join([x for x in j.keys() if 'lese3_' in x]) if bool([x for x in j.keys() if 'lese3_' in x]) else '-'
    lese3_ = j[lese3_] if lese3_ != '-' else '-'
    pls3_, rls3_, f1_ls3_, levd3_ = round(np.mean(lese3_['prec_lev'])*100, 2) if lese3_ != '-' else '-', \
                                    round(np.mean(lese3_['rec_lev'])*100, 2) if lese3_ != '-' else '-', \
                                    round(np.mean(lese3_['fs_lev'])*100, 2) if lese3_ != '-' else '-', \
                                    np.mean(lese3_['lev_d'])//3 if lese3_ != '-' else '-'
    print(f"|  {i.upper()}    |   {bleuscore_}   |   {bleuscore3_}  | {meteor_}  |  {p1_} |  {r1_}  |  {f1_1_}  |  {pl_} |  {rl_} |  {f1_l_} |  {pls_}  | {rls_}  |  {f1_ls_}  |  {levd1_}  |  {pls3_}  |  {rls3_}  |  {f1_ls3_}  |  {levd3_}  |   {ppl}  |")
print('+------------+-------------+-------------+-------------+-------+------+-----+--------+------+-----+-------+------+-----+-------------+-------+------+-----+---------------+---------------+')        

    
    
    
    
    
    



