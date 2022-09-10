from argparse import ArgumentParser
import numpy as np
import pandas as pd
import random, math
import networkx as nx
import logging
from dateutil.parser import parse

pd.options.mode.chained_assignment = None

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                    logging.FileHandler("generate_property_paths.log", mode='w'),
                    logging.StreamHandler()
                    ]
)

def WiDNeR(structured_triples):
    structured_triples_ = pd.read_csv(structured_triples, header=None, names=['head','relation', 'tail'], delimiter='\t')
    logging.info(f'reading {len(structured_triples_)} triples has been completed!')


    ## different relations - direct
    result_direct = structured_triples_.merge(structured_triples_, left_on='tail', right_on='head', how='inner')
    diff_rels_direct = result_direct.loc[result_direct['relation_x'] != result_direct['relation_y']]
    diff_rels_direct.drop_duplicates(inplace=True)
    diff_rels_direct = diff_rels_direct.groupby(['relation_x', 'relation_y']).size().to_frame('#direct').reset_index()

    ## different relations - shared_head
    result_shared_head = structured_triples_.merge(structured_triples_, left_on='head', right_on='head', how='inner')
    diff_rels_shared_head = result_shared_head.loc[result_shared_head['relation_x'] != result_shared_head['relation_y']]
    diff_rels_shared_head.drop_duplicates(inplace=True)
    diff_rels_shared_head = diff_rels_shared_head.groupby(['relation_x', 'relation_y']).size().to_frame('#shared_head').reset_index()

    ## different relations - shared_tail
    result_shared_tail = structured_triples_.merge(structured_triples_, left_on='tail', right_on='tail', how='inner')
    diff_rels_shared_tail = result_shared_tail.loc[result_shared_tail['relation_x'] != result_shared_tail['relation_y']]
    diff_rels_shared_tail.drop_duplicates(inplace=True)
    diff_rels_shared_tail = diff_rels_shared_tail.groupby(['relation_x', 'relation_y']).size().to_frame('#shared_tail').reset_index()

    ## combine different relations - direct, shared_head, and shared_tail
    diff_rels = diff_rels_direct.merge(diff_rels_shared_head, on=['relation_x', 'relation_y'], how='outer')
    diff_rels = diff_rels.merge(diff_rels_shared_tail, on=['relation_x', 'relation_y'], how='outer')
    diff_rels.fillna(0, inplace=True)


    ## same relations - direct
    same_rels_direct = result_direct.loc[(result_direct['relation_x'] == result_direct['relation_y']) & ((result_direct['head_x'] != result_direct['tail_x']) | (result_direct['head_x'] != result_direct['tail_y'])) ]
    same_rels_direct.drop_duplicates(inplace=True)
    same_rels_direct = same_rels_direct.groupby(['relation_x', 'relation_y']).size().to_frame('#direct').reset_index()


    ## same relations - shared_head
    same_rels_shared_head = result_shared_head.loc[(result_shared_head['relation_x'] == result_shared_head['relation_y']) & (result_shared_head['tail_x'] != result_shared_head['tail_y'])]
    same_rels_shared_head.drop_duplicates(inplace=True)
    same_rels_shared_head = same_rels_shared_head.groupby(['relation_x', 'relation_y']).size().to_frame('#shared_head').reset_index()
    same_rels_shared_head['#shared_head'] = same_rels_shared_head['#shared_head'].map(lambda x: x/2)


    ## same relations - shared_tail
    same_rels_shared_tail = result_shared_tail.loc[(result_shared_tail['relation_x'] == result_shared_tail['relation_y']) & (result_shared_tail['head_x'] != result_shared_tail['head_y'])]
    same_rels_shared_tail.drop_duplicates(inplace=True)
    same_rels_shared_tail = same_rels_shared_tail.groupby(['relation_x', 'relation_y']).size().to_frame('#shared_tail').reset_index()
    same_rels_shared_tail['#shared_tail'] = same_rels_shared_tail['#shared_tail'].map(lambda x: x/2)


    ## combine same relations - direct, shared_head, and shared_tail
    same_rels = same_rels_direct.merge(same_rels_shared_head, on=['relation_x', 'relation_y'], how='outer')
    same_rels = same_rels.merge(same_rels_shared_tail, on=['relation_x', 'relation_y'], how='outer')
    same_rels.fillna(0, inplace=True)

    ## combine different and same relations
    diff_same = pd.concat([diff_rels, same_rels], ignore_index=True)

    diff_same['weight'] = diff_same['#direct'] + diff_same['#shared_head'] + diff_same['#shared_tail']

    rel_rel_net = diff_same.drop(columns=['#direct', '#shared_head', '#shared_tail'])

    return rel_rel_net



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('command', choices=['WiDNeR'])
    parser.add_argument('--structured_triples', help='Input the file with structured triples')

    args = parser.parse_args()

    if args.command == 'WiDNeR':
        rel_rel_net = WiDNeR(structured_triples=args.structured_triples)
        rel_rel_net_.rename(columns={'relation_x': 'source', 'relation_y': 'target'}, inplace=True)
        rel_rel_net.to_csv(f'{args.structured_triples}_WiDNeR')
