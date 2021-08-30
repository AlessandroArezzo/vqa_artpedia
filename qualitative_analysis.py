import csv
import pandas as pd
import numpy as np

artpedia_correct_path = "./data/output/correct_pred_artpedia.csv"
artpedia_error_path = "./data/output/error_pred_artpedia.csv"
artpedia_dt_correct_path = "./data/output/correct_pred_artpedia_dt.csv"
artpedia_dt_error_path = "./data/output/error_pred_artpedia_dt.csv"

def read_data_from_csv(csv_path):
    data = {}
    with open(csv_path, 'r') as data_file:
        reader = csv.reader(data_file)
        for idx, row in enumerate(reader):
            if idx > 0:
                image_idx = row[0]
                question_token = row[1]
                question = row[2]
                answer = row[3]
                pred = row[4]
                if image_idx not in data.keys():
                    data[image_idx] = {}
                    data[image_idx]["questions_token"] = []
                    data[image_idx]["questions"] = []
                    data[image_idx]["answers"] = []
                    data[image_idx]["preds"] = []
                data[image_idx]["questions_token"].append(question_token)
                data[image_idx]["questions"].append(question)
                data[image_idx]["answers"].append(answer)
                data[image_idx]["preds"].append(pred)
    return data

artpedia_correct_data = read_data_from_csv(artpedia_correct_path)
artpedia_error_data = read_data_from_csv(artpedia_error_path)
artpedia_dt_correct_data = read_data_from_csv(artpedia_dt_correct_path)
artpedia_dt_error_data = read_data_from_csv(artpedia_dt_error_path)
result_path = "./data/output/correct_only_dt"
out_df_results = pd.DataFrame(columns=['idx_image', 'question', 'answer', 'artpedia_dt_pred', 'artpedia_error'])
for image_dt_idx in artpedia_dt_correct_data.keys():
    questions = artpedia_dt_correct_data[image_dt_idx]["questions_token"]
    for q_idx, q in enumerate(questions):
        if image_dt_idx not in artpedia_correct_data.keys() \
                or q not in artpedia_correct_data[image_dt_idx]["questions_token"]:
            try:
                for idx, question in enumerate(artpedia_error_data[image_dt_idx]["questions_token"]):
                    if question == q:
                        data_to_add = [image_dt_idx, artpedia_dt_correct_data[image_dt_idx]["questions"][q_idx],
                                       artpedia_dt_correct_data[image_dt_idx]["answers"][q_idx],
                                       artpedia_dt_correct_data[image_dt_idx]["preds"][q_idx],
                                       artpedia_error_data[image_dt_idx]["preds"][idx]]
                        data_df_scores = np.hstack((np.array(data_to_add).reshape(1, -1)))
                        out_df_results = out_df_results.append(pd.Series(data_df_scores.reshape(-1),
                                                                     index=out_df_results.columns), ignore_index=True)
                out_df_results.to_csv(result_path, index=False, header=True)
            except KeyError:
                continue

