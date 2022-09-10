from argparse import ArgumentParser
import numpy as np
import pandas as pd
import random, math
import networkx as nx
import extract_labels
import logging, tqdm
import os


def add_bool_arg(parser, name, help, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, help=help, dest=name, action='store_true')
    group.add_argument('--no-' + name, help=help, dest=name, action='store_false')
    parser.set_defaults(**{name:default})


def config_logger(dir):
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                    logging.FileHandler(os.path.join(dir, "create_dataset_g.log"), mode='w'),
                    logging.StreamHandler()
                    ]
    )
    #logger = logging.getLogger(__name__)
    #logging.getLogger().setLevel(logging.DEBUG)
    #return logger


def freq_entities(triples):
    heads = triples.loc[:, ['head']]
    tails = triples.loc[:, ['tail']]
    heads.rename(columns={'head' :'entity'},inplace=True)
    tails.rename(columns={'tail' :'entity'},inplace=True)
    ents_freqs = pd.concat([heads, tails])['entity'].value_counts().rename_axis('entity').reset_index(name='freq')
    return ents_freqs

def create_dataset(triples, relation_types, entity_types, rel_train_size, rel_valid_size, rel_test_size, ent_train_size, ent_valid_size,
                                ent_test_size, remove_duplicate_relations, remove_inverse_relations, random_seed, extract_relations_textual_info, extract_entities_textual_info, rel_freq, k_train,
                                k_valid, k_test, dir):
    ## read the triples from a file
    triples_ = pd.read_csv(triples, header=None, names=['head','relation', 'tail'], delimiter='\s+')
    logging.info(f'reading {len(triples_)} input triples from the file "{triples}" has been completed!')

    ## remove relations which occur n=2 or less number of times - to avoid wrong triples.
    relations_freq = triples_.groupby(['relation']).size().reset_index(name='total_count') #.sort_values(['total_count'], ascending=False)
    relations_to_remove = relations_freq.loc[relations_freq['total_count'] < rel_freq]['relation'].tolist()
    triples_ = triples_.loc[~triples_['relation'].isin(relations_to_remove)]
    logging.info(f'{len(relations_to_remove)} relations occur less than {rel_freq} times and hence, have been removed! which reduces the triples to {len(triples_)}')

    if remove_inverse_relations:
        ##remove inverse relations
        relations_to_remove = remove_inverse_relations(triples_)
        triples_ = triples_.loc[~triples_['relation'].isin(relations_to_remove)]
        logging.info(f'{len(relations_to_remove)} inverse relations removed: {len(triples_)} triples left')

    ## extract labels, descriptions, and aliases and keep only those entities which have labels
    if extract_entities_textual_info:
        entities = set(triples['head'].tolist())
        entities.update(triples['tail'].tolist())

        entities_with_labels = extract_labels.extract_entity_labels(entities)
        triples_ = triples_.loc[triples_['head'].isin(entities_with_labels) & triples_['tail'].isin(entities_with_labels)]
        logging.info(f'extracting labels, aliases, and descriptions of entities finished and only {len(entities_with_labels)} have labels!')

    ##  extract labels, descriptions, and aliases and keep only those relations which have labels
    if extract_relations_textual_info:
        relations_with_labels = extract_labels.extract_relation_labels(list(set(triples_['relation'].tolist())))
        triples_ = triples_.loc[triples_['relation'].isin(relations_with_labels)] ## No label found for the relation P5326 becuase the it has been deleted from Wiidata
        logging.info(f'extracting labels, aliases, and descriptions of relations finished and only {len(relations_with_labels)} have labels!')

    ## remove duplicate relations
    if remove_duplicate_relations:
        relations_to_remove = duplicate_relations(triples_)
        triples_ = triples_.loc[~triples_['relation'].isin(relations_to_remove)]
        logging.info(f'{relations_to_remove} relations are duplicate and thus, have been removed!')


    if extract_relations_textual_info or remove_duplicate_relations or extract_entities_textual_info or remove_inverse_relations:
        triples_.to_csv(os.path.join(dir, f'{triples}_preprocessed.tsv'), sep=' ', index=False, header=False)

    ## read relation types from a file
    relation_types_ = pd.read_csv(relation_types, header=None, names=['relation','type'],delimiter=' ')
    relation_types_ = relation_types_.loc[relation_types_['relation'].isin(triples_['relation'].tolist())]


    ## for those entities which do not have type info add 'None' as a type
    relation_wout_type = triples_.loc[~triples_['relation'].isin(relation_types_['relation'].tolist()), 'relation'].drop_duplicates().to_frame()
    relation_wout_type['type'] = 'None'
    relation_types_ = pd.concat([relation_types_,relation_wout_type])
    logging.info(f'{len(relation_wout_type)} relations do not have type info in "{relation_types}"! and hence assigned "None"')

    ## split relations into train, valid, and test set relations
    train_rels, valid_rels, test_rels = split_relations_diff_dom(relation_types_, rel_train_size, rel_valid_size, rel_test_size, random_seed, dir)
    logging.info(f'relations splited to {len(train_rels)} train relations, {len(valid_rels)} validation relations, and {len(test_rels)} test relations!')

    ## split triples into train, valid, and test sets
    split_triples(train_rels, valid_rels, test_rels, triples_, entity_types, ent_train_size, ent_valid_size, ent_test_size, k_train, k_valid, k_test, random_seed, dir)



def split_relations(relation_types_, rel_train_size, rel_valid_size, rel_test_size, random_seed):
    ###### one option to consider instead of randomly splitting the relations is to take most popular ones for training and less popular ones for valid and test
    random.seed(random_seed)

    train_rels = []
    valid_rels = []
    test_rels = []
    while not relation_types_.empty:
        type, count = compute_freq(relation_types_, column_to_count='type')
        train_number = round(count*rel_train_size)
        valid_number = round(count*rel_valid_size)

        relations = relation_types_.loc[relation_types_['type'] == type]['relation'].tolist()
        relation_types_ = relation_types_[~relation_types_['relation'].isin(relations)]

        train_rel = random.sample(relations, k=train_number)
        relations = [rel for rel in relations if rel not in train_rel]

        if(len(relations) < valid_number):
            valid_rel = relations
        else:
            valid_rel = random.sample(relations, k=valid_number)

        test_rel = [rel for rel in relations if rel not in valid_rel]

        train_rels.extend(train_rel)
        valid_rels.extend(valid_rel)
        test_rels.extend(test_rel)

    f1 = open(os.path.join(dir,"train_rels.txt"), "w")
    f2 = open(os.path.join(dir,"valid_rels.txt"), "w")
    f3 = open(os.path.join(dir,"test_rels.txt"), "w")
    for element in train_rels:
        f1.write(element + "\n")
    f1.close()
    for element in valid_rels:
        f2.write(element + "\n")
    f2.close()
    for element in test_rels:
        f3.write(element + "\n")
    f3.close()


    return train_rels, valid_rels, test_rels

def split_relations_diff_dom(relation_types_, rel_train_size, rel_valid_size, rel_test_size, random_seed, dir):
    random.seed(random_seed)

    train_rels = []
    valid_rels = []
    test_rels = []

    total_rel_count = len(set(relation_types_['relation'].tolist()))
    logging.info(f'Total number of relations are: {total_rel_count}')

    train_number = round(total_rel_count*rel_train_size)
    valid_number = round(total_rel_count*rel_valid_size)

    while not relation_types_.empty:
        type, count = compute_freq(relation_types_, column_to_count='type')

        relations = relation_types_.loc[relation_types_['type'] == type]['relation'].tolist()
        relation_types_ = relation_types_[~relation_types_['relation'].isin(relations)]

        if (len(train_rels) < train_number):
            train_rels.extend(relations)

        elif (len(valid_rels) < valid_number):
            valid_rels.extend(relations)
        else:
            test_rels.extend(relations)

    return train_rels, valid_rels, test_rels

def compute_freq(data, column_to_count):
    freq = data[column_to_count].value_counts().rename_axis('unique_values').reset_index(name='counts')#.iloc[0]
    return freq['unique_values'].iloc[0], freq['counts'].iloc[0]

def sample_entities_type(triples, entity_types, fraction, random_seed):
    entities = set(triples['head'].tolist())
    entities.update(triples['tail'].tolist())
    entity_types_=pd.read_csv(entity_types,delimiter=' ')
    entity_types_=entity_types_.loc[entity_types_['entity'].isin(entities)]
    sampled_entities=entity_types_.groupby(['type'])['entity'].sample(frac=fraction, random_state=random_seed, replace=False).drop_duplicates().tolist()

    return sampled_entities

def split_triples(train_rels, valid_rels, test_rels, triples, entity_types, ent_train_size, ent_valid_size, ent_test_size, k_train, k_valid, k_test, random_seed, dir):
    ## get training triples
    logging.info(f'========================== Getting training triples ==================')
    train_triples = triples.loc[triples['relation'].isin(train_rels)]
    logging.info(f'{len(train_triples)} found with the {len(train_rels)} train relations')

    train_triples = create_kcore(train_triples, k_train)
    logging.info(f'{len(train_triples)} training triples with core {k_train} ')
    #remove skewed relations
    relations_to_remove_head = remove_skewed_relations(train_triples, 'head')
    relations_to_remove_tail = remove_skewed_relations(train_triples, 'tail')
    train_triples = train_triples.loc[~train_triples['relation'].isin(relations_to_remove_head + relations_to_remove_tail)]
    logging.info(f'{len(train_triples)} training triples after removing skewed relations -- with {len(set(train_triples["relation"].tolist()))} relations')

    train_entities = set(train_triples['head'].tolist())
    train_entities.update(train_triples['tail'].tolist())
    logging.info(f'{len(train_entities)} training entities')
    train_triples.to_csv(os.path.join(dir,'ind-train.tsv'), sep='\t', index=False, header=False)
    train_entities.to_csv(os.path.join(dir,'ents_train.txt'), sep='\t', index=False, header=False)

    #get valid triples
    logging.info(f'========================== Getting validation triples ==================')
    #makes sure only relations are unseen. entities could be either seen or unseen.
    valid_triples = triples.loc[triples['relation'].isin(valid_rels)]

    print("valid_triples=", len(valid_triples))

    valid_triples = create_kcore(valid_triples, k_valid)
    print("valid_triples_core=", len(valid_triples))
    #remove skewed relations
    relations_to_remove_head = remove_skewed_relations(valid_triples, 'head')
    relations_to_remove_tail = remove_skewed_relations(valid_triples, 'tail')
    valid_triples = valid_triples.loc[~valid_triples['relation'].isin(relations_to_remove_head + relations_to_remove_tail)]
    logging.info(f'{len(valid_triples)} validation triples after removing skewed relations -- with {len(set(valid_triples["relation"].tolist()))} relations')

    valid_entities = set(valid_triples['head'].tolist())
    valid_entities.update(valid_triples['tail'].tolist())
    logging.info(f'{len(valid_entities)} validation entities')
    valid_triples.to_csv(os.path.join(dir,'ind-dev.tsv'), sep='\t', index=False, header=False)
    valid_entities.to_csv(os.path.join(dir,'ents_dev.txt'), sep='\t', index=False, header=False)


    ### how many new entities in validation set
    new_entities = valid_entities - train_entities
    logging.info(f'Number of entities in validation set but not in train sets: {len(new_entities)}')

    #get test triples
    logging.info(f'========================== Getting test triples ==================')
    #makes sure only relations are unseen. entities could be either seen or unseen.
    test_triples = triples.loc[triples['relation'].isin(test_rels)]

    test_triples = create_kcore(test_triples, k_test)
    print("test_triples_core=", len(test_triples))

    #remove skewed relations
    relations_to_remove_head = remove_skewed_relations(test_triples, 'head')
    relations_to_remove_tail = remove_skewed_relations(test_triples, 'tail')
    test_triples = test_triples.loc[~test_triples['relation'].isin(relations_to_remove_head + relations_to_remove_tail)]
    logging.info(f'{len(test_triples)} test triples after removing skewed relations -- with {len(set(test_triples["relation"].tolist()))} relations')

    test_entities = set(test_triples['head'].tolist())
    test_entities.update(test_triples['tail'].tolist())
    logging.info(f'{len(test_entities)} test entities')
    test_triples.to_csv(os.path.join(dir,'ind-test.tsv'), sep='\t', index=False, header=False)
    test_entities.to_csv(os.path.join(dir,'ents_test.txt'), sep='\t', index=False, header=False)

    new_entities = (test_entities - train_entities) - valid_entities
    logging.info(f'Number of entities in test set but not in train and validation sets: {len(new_entities)}')

    ## save relations
    train_rels = set(train_triples['relation'].tolist())
    valid_rels = set(valid_triples['relation'].tolist())
    test_rels = set(test_triples['relation'].tolist())

    f1 = open(os.path.join(dir,"train_rels.txt"), "w")
    f2 = open(os.path.join(dir,"valid_rels.txt"), "w")
    f3 = open(os.path.join(dir,"test_rels.txt"), "w")
    for element in train_rels:
        f1.write(element + "\n")
    f1.close()
    for element in valid_rels:
        f2.write(element + "\n")
    f2.close()
    for element in test_rels:
        f3.write(element + "\n")
    f3.close()

def create_kcore(train_triples, k):
    ents_freqs = freq_entities(train_triples)
    ent_to_remove = ents_freqs[ents_freqs.freq < k]['entity'].tolist()
    while len(ent_to_remove) > 0:
        train_triples=train_triples[~train_triples['head'].isin(ent_to_remove) & ~train_triples['tail'].isin(ent_to_remove)]
        ents_freqs = freq_entities(train_triples)
        ent_to_remove = ents_freqs[ents_freqs.freq < k]['entity'].tolist()
    return train_triples

def generate_finegrained_types(entity_types):
    entity_types_ = pd.read_csv(entity_types, header=None,names=['entity', 'type', 'num_instances'], delimiter=' ')
    idx= entity_types_.groupby(['entity'])['num_instances'].transform(min) == entity_types_['num_instances']
    entity_types_final = entity_types_[idx]
    print(entity_types_final)
    entity_types_final.to_csv('entity_types_final.txt', sep=' ', index=False)

def remove_skewed_relations(triples, entity):
    triples_ = triples.groupby(['relation', entity]).size().reset_index(name='count')
    idx= triples_.groupby(['relation'])['count'].transform(max) == triples_['count']
    triples_=triples_[idx]
    relations_freq= triples.groupby(['relation']).size().reset_index(name='total_count')
    triples_ = triples_[idx]
    triples_ = pd.merge(triples_, relations_freq, how='inner', on='relation')
    triples_['skewed']= triples_['count'] / triples_['total_count']
    relations_to_remove= triples_.loc[(triples_['skewed'] >= 0.5)]['relation'].tolist()

    return relations_to_remove

def duplicate_relations(triples):
    relations_freq = triples.groupby(['relation']).size().reset_index(name='total_count')
    relations_freq_dict = dict(zip(relations_freq.relation, relations_freq.total_count))
    relations = list(set(triples['relation'].tolist()))
    relations_to_remove = []
    for i in range(len(relations)):
        for j in range(i+1, len(relations)):
            maximum = max(relations_freq_dict[relations[i]], relations_freq_dict[relations[j]])
            minimum = min(relations_freq_dict[relations[i]], relations_freq_dict[relations[j]])
            if (maximum / 2 ) < minimum:
                rij = triples.loc[(triples['relation']==relations[i]) | (triples['relation']==relations[j])][['head', 'tail']].value_counts().reset_index(name='count')
                common= len(rij.loc[rij['count']>1])
                ri_common= common / relations_freq_dict[relations[i]]
                rj_common= common / relations_freq_dict[relations[j]]
                if ( ri_common > 0.5) |(rj_common > 0.5):
                    if relations_freq_dict[relations[i]] <= relations_freq_dict[relations[j]]:
                        relations_to_remove.append(relations[i])
                    else:
                        relations_to_remove.append(relations[j])
                    print(relations[i], relations[j])
    return relations_to_remove

def remove_inverse_relations(triples):
    relations_to_remove = []
    inverse_pairs = pd.read_csv('dataset_creation/inputs/inverse_props_short.txt', header=None, names=['r1','r2'], delimiter='\s+')
    inverse_pairs_checked = []
    for index, row in inverse_pairs.iterrows():
        if [row['r2'], row['r1']] not in inverse_pairs_checked:
            num_r1 = len(triples.loc[triples['relation'] == row['r1']])
            num_r2 = len(triples.loc[triples['relation'] == row['r2']])
            if num_r1  > num_r2:
                relations_to_remove.append(row['r1'])
            else:
                relations_to_remove.append(row['r2'])
            inverse_pairs_checked.append([row['r2'], row['r1']])

    return relations_to_remove


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('command', choices=['generate_finegrained_types', 'create_dataset'])
    parser.add_argument('--triples', help='Input the file with triples')
    parser.add_argument('--relation_types', help='Input the file with types of relations (properties) in wikidata')
    parser.add_argument('--entity_types', help='Input the file with classes of entities in wikidata',default=None)
    parser.add_argument('--rel_train_size', type=float, help='how many percentage of the'
                                            ' triples', default=0.8)
    parser.add_argument('--rel_valid_size', type=float, help='how many percentage of the'
                                              ' relations should be in the'
                                              'valid triples', default=0.1)
    parser.add_argument('--rel_test_size', type=float, help='how many percentage of the'
                                              ' relations should be in the'
                                              'test triples', default=0.1)
    parser.add_argument('--ent_train_size', type=float, help='how many percentage of the'
                                            'entities should be in the training'
                                            ' triples', default=0.5)
    parser.add_argument('--ent_valid_size', type=float, help='how many percentage of the'
                                              ' entities should be in the'
                                              'valid triples', default=0.9)
    parser.add_argument('--ent_test_size', type=float, help='how many percentage of the'
                                              ' entities should be in the'
                                              'test triples', default=0.1)
    parser.add_argument('--seed', type=int, help='Random seed', default=11111)
    parser.add_argument('--rel_freq', type=int, help='The minimum number of triples every relation should have', default=3)
    parser.add_argument('--k_train', type=int, help='The k value for training core ', default=5)
    parser.add_argument('--k_valid', type=int, help='The k value for validation core ', default=4)
    parser.add_argument('--k_test', type=int, help='The k value for test core ', default=4)
    parser.add_argument('--dir', help='The name of the directory to store results')

    add_bool_arg(parser, 'remove_duplicate_relations', 'Enter True if duplicate relations should be removed')
    add_bool_arg(parser, 'remove_inverse_relations', 'Enter True if inverse relations should be removed')
    add_bool_arg(parser, 'extract_relations_textual_info', 'Extract labels, aliases, and descriptions of relations from Wikidata')
    add_bool_arg(parser, 'extract_entities_textual_info', 'Extract labels, aliases, and descriptions of entities from Wikidata')

    args = parser.parse_args()

    if args.command == 'generate_finegrained_types':
        generate_finegrained_types(args.entity_types)

    print(os.path.join(args.dir,'ind-train.tsv'))
    if args.command == 'create_dataset':
        if not os.path.exists(args.dir):
            os.makedirs(args.dir)
        else:
            raise ValueError('Directory already exists.')

        #logger =
        config_logger(args.dir)

        logging.info(f'ent_train_size={args.ent_train_size} ent_valid_size={args.ent_valid_size} ent_test_size={args.ent_test_size}\
        rel_train_size={args.rel_train_size} rel_valid_size={args.rel_valid_size} rel_test_size={args.rel_test_size} \
        remove_duplicate_relations={args.remove_duplicate_relations} remove_inverse_relations={args.remove_inverse_relations} \
        random_seed={args.seed} \
        extract_relations_textual_info={args.extract_relations_textual_info} \
        extract_entities_textual_info={args.extract_entities_textual_info} \
        rel_freq={args.rel_freq} k_train={args.k_train} k_valid={args.k_valid} k_test={args.k_test}')

        create_dataset(triples=args.triples,
                    relation_types=args.relation_types,
                    entity_types=args.entity_types,
                    ent_train_size=args.ent_train_size,
                    ent_valid_size=args.ent_valid_size,
                    ent_test_size=args.ent_test_size,
                    rel_train_size=args.rel_train_size,
                    rel_valid_size=args.rel_valid_size,
                    rel_test_size=args.rel_test_size,
                    remove_duplicate_relations=args.remove_duplicate_relations,
                    remove_inverse_relations =args.remove_inverse_relations,
                    random_seed=args.seed,
                    extract_relations_textual_info=args.extract_relations_textual_info,
                    extract_entities_textual_info=args.extract_entities_textual_info,
                    rel_freq=args.rel_freq,
                    k_train=args.k_train,
                    k_valid=args.k_valid,
                    k_test =args.k_test,
                    dir=args.dir)
