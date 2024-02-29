import argparse
import glob
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap

from encoder.model import SpeakerEncoder
from encoder.visualizations import colormap

user_ids = {
    2: 'Alex',
    3: 'Adrian',
    5: 'Liam',
    7: 'Conor',
    6: 'Richard',
    12: 'Domhnall'
}
user_id_accuracies = {
    'shane': [26.666666666666668, 40.0, 51.282051282051285],
    'Domhnall': [92.77777777777777, 99.44444444444444, 100.0],
    'Richard': [78.5, 87.0, 91.5],
    'gregory': [23.68421052631579, 40.78947368421053, 52.63157894736842],
    'Liam': [84.0, 93.0, 93.5],
}

# visual-dtw imports
sys.path.append('/home/domhnall/Repos/visual-dtw/app')
from main.research.research_utils import create_templates, preprocess_signal, get_accuracy
from main.utils.io import read_pickle_file, read_json_file, read_ark_file
from main.utils.parsing import extract_template_info


def main(args):
    # get default templates w/o preprocessing
    all_phrases = read_json_file('/home/domhnall/Repos/visual-dtw/data/phrases.json')
    template_lookup = {}
    for template in glob.glob(os.path.join('/home/domhnall/Repos/visual-dtw/data/AE_norm_2', '*.ark')):
        basename = os.path.basename(template).replace('.ark', '')
        user_id, phrase_set, phrase_id, session_id = extract_template_info(basename)
        key = f'AE_norm_2_{user_id}_{phrase_set + phrase_id}_{session_id}'
        template_lookup[key] = template
    default_templates = {}
    for template_id in np.genfromtxt('/home/domhnall/Repos/visual-dtw/data/default_templates.np',
                                     delimiter=',', dtype='str'):
        user_id, phrase_set, phrase_id, session_id = extract_template_info(template_id, from_default_list=True)
        template_path = template_lookup[template_id]
        blob = read_ark_file(template_path)
        phrase = all_phrases[phrase_set][phrase_id]

        user_templates = default_templates.get(user_id, [])
        user_templates.append((phrase, blob))
        default_templates[user_id] = user_templates
    num_default_templates = sum([len(v) for k, v in default_templates.items()])
    print('Num default templates:', num_default_templates)
    for k, v in default_templates.items():
        print(k, 'has', len(v), 'templates')

    # get non-default templates w/o preprocessing
    non_default_video_dirs = []
    non_default_templates = {}
    for video_dir in args.non_default_video_dirs:
        if not video_dir:
            continue

        if '*' in video_dir:
            sub_dirs = glob.glob(video_dir + '/')
            random.shuffle(sub_dirs)
            count = 0
            for dir in sub_dirs:
                videos = glob.glob(os.path.join(dir, '*.mp4'))
                if len(videos) > 20:
                    non_default_video_dirs.append(dir)
                    count += 1
                if count == args.count:
                    break
        else:
            non_default_video_dirs.append(video_dir)
    print('Num non-default users:', len(non_default_video_dirs))
    for video_dir in non_default_video_dirs:
        templates = create_templates(video_dir,
                                     regexes=['PV0*(\d+)_P10*(\d+)_S0*(\d+)'],
                                     phrase_lookup=all_phrases['PAVA-DEFAULT'],
                                     phrase_column='Groundtruth',
                                     debug=True,
                                     preprocess=False,
                                     save=True)
        if len(templates) == 0:
            continue
        non_default_templates[Path(video_dir).name] = templates

    # combine default w/ non-default
    all_templates = {**default_templates, **non_default_templates}

    # load speaker encoder model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpeakerEncoder(device, loss_device='cpu')
    models_dir = Path('encoder/saved_models')
    state_path = models_dir.joinpath(args.run_id + '.pt')
    checkpoint = torch.load(state_path)
    model.load_state_dict(checkpoint['model_state'])

    # get encoder embeddings
    all_embeds = []
    amounts = []
    classes = []
    for i, (_id, templates) in enumerate(all_templates.items()):
        try:
            _id = int(_id)
            _id = user_ids[_id]
        except ValueError:
            pass
        classes.append(_id)
        amounts.append(len(templates))
        for phrase, template in templates:
            template = template[np.newaxis, ...]
            _input = torch.from_numpy(template).to(device)
            embeds = model(_input)
            embeds = embeds.detach().cpu().numpy()
            all_embeds.append(embeds[0])
    all_embeds = np.asarray(all_embeds)
    print('All embeds:', all_embeds.shape)

    # plot embeddings via UMAP projection on scatterplot
    # umap is stochastic so running it gives a different graph each time
    # set seed to stop this from happening
    reducer = umap.UMAP(random_state=2021)
    projected = reducer.fit_transform(all_embeds)
    projected = projected.tolist()
    start = 0
    scatters = []
    default_projections = {}
    other_projections = {}
    for i, (_id, amount) in enumerate(zip(classes, amounts)):
        end = start + amount
        projection = projected[start:end]
        if _id in ['Alex', 'Adrian', 'Conor']:
            default_projections[_id] = projection
        else:
            other_projections[_id] = projection
        projection = np.asarray(projection)
        scatter = plt.scatter(projection[:, 0], projection[:, 1], color=colormap[i])
        scatters.append(scatter)
        start += amount

    print('Num Default Projections:', sum([len(v) for k, v in default_projections.items()]))

    # find centroids for every cluster of projections
    def get_centroids(_projections):
        projection_centroids = {}
        for _id, projections in _projections.items():
            num_projections = len(projections)
            all_avs = []
            for i in range(num_projections):
                total = 0
                for j in range(num_projections):
                    if i == j:
                        continue
                    p1, p2 = projections[i], projections[j]
                    dist = np.linalg.norm(np.asarray(p1) - np.asarray(p2))
                    total += dist
                av = total / (num_projections - 1)
                all_avs.append(av)
            min_index = all_avs.index(min(all_avs))
            projection_centroids[_id] = projections[min_index]

        return projection_centroids

    default_projection_centroids = get_centroids(default_projections)
    non_default_projection_centroids = get_centroids(other_projections)

    # plot centroids and lines between them
    centroid_projections = np.asarray(list(default_projection_centroids.values())
                                      + list(non_default_projection_centroids.values()))
    scatter = plt.scatter(centroid_projections[:, 0], centroid_projections[:, 1], c='black', marker='x')
    scatters.append(scatter)
    classes.append('Centroids')
    default_projection_centroid_points = list(default_projection_centroids.values())
    for i in range(len(default_projection_centroid_points)-1):
        for j in range(i+1, len(default_projection_centroid_points)):
            p1 = default_projection_centroid_points[i]
            p2 = default_projection_centroid_points[j]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c='black')

    # find nearest from each person to default clusters
    sorted_user_dists = {}
    unsorted_user_dists = {}
    default_projection_centroids = list(default_projection_centroids.items())
    for _id, non_default_centroid_projection in non_default_projection_centroids.items():
        min_d = 10000
        min_centroid = None
        all_ds = []
        for d_id, default_centroid_projection in default_projection_centroids:
            dist = np.linalg.norm(np.asarray(non_default_centroid_projection) - np.asarray(default_centroid_projection))
            all_ds.append(dist)
            if dist < min_d:
                min_d = dist
                min_centroid = d_id
        # summary = (0.7 * min_d) + (0.3 * np.mean(all_ds))
        # summary = sum(all_ds)
        # print(_id, 'closest to', min_centroid, 'w/ distance', min_d, 'w/ mean', np.mean(all_ds), 'summary', summary)
        print(_id, 'closest to', min_centroid, 'w/ distance', min_d)
        # user_dists[_id] = summary
        sorted_user_dists[_id] = sorted(all_ds)
        unsorted_user_dists[_id] = all_ds

    # plot graph of performance vs distance summary to default model
    default_templates = [
        (p, preprocess_signal(t))
        for k, v in default_templates.items()
        for p, t in v
    ]
    non_default_templates = {
        k: [(p, preprocess_signal(t)) for p, t in v]
        for k, v in non_default_templates.items()
    }
    weights = [0.6, 0.3, 0.1]
    user_accuracies = {}
    for user_id, user_templates in non_default_templates.items():
        try:
            user_id = int(user_id)
            user_id = user_ids[user_id]
        except ValueError:
            pass

        accuracies = user_id_accuracies.get(user_id, None)
        if not accuracies:
            accuracies = get_accuracy(default_templates, user_templates, debug=True)[0]

        accuracy_summary = sum([accuracy * weight for accuracy, weight in zip(accuracies, weights)])
        user_accuracies[user_id] = accuracy_summary

        # add accuracies as text to the non-centroids scatter graph
        centroid_projection = non_default_projection_centroids[user_id]
        plt.annotate(f'{round(accuracy_summary)}', (centroid_projection[0]-0.5, centroid_projection[1]-0.5))

    plt.legend(scatters, classes)
    plt.show()

    # xs, ys = [], []
    # for user_id, dist in user_dists.items():
    #     xs.append(dist)
    #     ys.append(user_accuracies[user_id])
    # plt.scatter(xs, ys)
    # plt.ylim((0, 100))
    # # plt.xlim((0, max(xs)))
    # plt.xlabel('Distance')
    # plt.ylabel('Accuracy')
    # plt.show()

    # # plot graph of sorted default centroid dists and accuracy
    # ds = [[], [], []]
    # acs = []
    # u_ids = []
    # for user_id, dists in sorted_user_dists.items():
    #     for i in range(3):
    #         ds[i].append(dists[i])
    #     accuracy = user_accuracies[user_id]
    #     acs.append(accuracy)
    #     u_ids.append(user_id)
    #     print(user_id, dists, accuracy)
    #
    # width = 0.35
    # df = pd.DataFrame({
    #     '1st closest': ds[0],
    #     '2nd closest': ds[1],
    #     '3rd closest': ds[2],
    #     'accuracy': acs
    # })
    # ax1 = df[['1st closest', '2nd closest', '3rd closest']].plot(kind='bar', width=width)
    # ax2 = df['accuracy'].plot(secondary_y=True, label='accuracy', style='.-', c='black')
    # ax1.set_ylabel('Distance')
    # ax2.set_ylabel('Accuracy')
    # ax2.set_ylim((0, 100))
    # ax2.legend(loc=0)
    # plt.xticks(np.arange(len(u_ids)), u_ids, rotation=45)
    # plt.xlim([-width, len(df['3rd closest']) - width])
    # plt.show()

    # plot graph of unsorted default centroid distances and accuracy
    ds = [[], [], []]
    acs = []
    u_ids = []
    for user_id, dists in unsorted_user_dists.items():
        for i in range(3):
            ds[i].append(dists[i])
        accuracy = user_accuracies[user_id]
        acs.append(accuracy)
        u_ids.append(user_id)
        print(user_id, dists, accuracy)

    width = 0.35
    default_centroid_names = [c[0] for c in default_projection_centroids]
    df = pd.DataFrame({
        default_centroid_names[0]: ds[0],
        default_centroid_names[1]: ds[1],
        default_centroid_names[2]: ds[2],
        'accuracy': acs
    })
    ax1 = df[default_centroid_names].plot(kind='bar', width=width)
    ax2 = df['accuracy'].plot(secondary_y=True, label='accuracy', style='.-', c='black')

    ax1.set_ylabel('Distance')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim((0, 100))
    ax2.legend(loc=0)

    plt.xticks(np.arange(len(u_ids)), u_ids, rotation=45)
    plt.xlim([-width, len(df[default_centroid_names[2]]) - width])
    plt.show()

    """
    python encoder_show_umap.py voxceleb_all /media/alex/Storage/Domhnall/datasets/sravi_dataset/pava_users/shane,/media/alex/Storage/Domhnall/datasets/sravi_dataset/liopa/pava/1_initial/12,/media/alex/Storage/Domhnall/datasets/sravi_dataset/liopa/pava/1_initial/6,/media/alex/Storage/Domhnall/datasets/sravi_dataset/pava_users/gregory,/media/alex/Storage/Domhnall/datasets/sravi_dataset/liopa/pava/1_initial/5,/media/alex/Storage/Domhnall/datasets/sravi_dataset/groundtruth/pava_groundtruth_export_2021-02-24--10:28:58/a7491ac4-d193-4c90-9490-857971e13bfe,/media/alex/Storage/Domhnall/datasets/sravi_dataset/groundtruth/pava_groundtruth_export_2021-02-24--10:28:58/192d13c5-8f76-4c5d-8f31-0d7a126d56e4
    
    Observations:
    
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_id')
    parser.add_argument('non_default_video_dirs', type=lambda s: s.split(','))
    parser.add_argument('--count', type=int, default=5)

    args = parser.parse_args()

    main(args)
