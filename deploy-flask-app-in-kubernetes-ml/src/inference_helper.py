from import_lib_helper import *


def get_path(country):
    """get path of assets"""
    model_path = f"models/actv/release-02-user-segment-level-nu/{country}/gbr_us_segment_level_actv_new_user_completion_rate_05_10_2021.pkl"
    feature_path = (
        f"models/actv/release-02-user-segment-level-nu/{country}/features.json"
    )
    inventory_path = f"inventory/{country}_segment_level_actv.pkl"
    return model_path, feature_path, inventory_path


def read_pickle(path):
    """reads pickle file"""
    return pd.read_pickle(path)


def read_json(path):
    """reads json file"""
    return pd.read_json(path)


def gen_prediction(model, pred_feat):
    """predicts outcomes using the trained model"""
    result = model.predict(pred_feat)
    return result


def projects_rank_gen(pred_feat):
    """ranking projects according to score"""
    pred_feat.sort_values(
        by=["rank_score", "completion_rate"],
        axis=0,
        inplace=True,
        ascending=[False, False],
    )
    pred_feat.set_index(np.arange(pred_feat.shape[0]), inplace=True)
    return pred_feat


def create_final_response(country, ranked_projects):
    """"creates final response to return in API request"""
    response = {}
    projects = ranked_projects[
        ["project_id", "rank_score", "updated_at", "difficulty_score", "provider"]
    ]

    projects.insert(2, column="ds_rank", value=np.arange(ranked_projects.shape[0]))
    response["projects"] = projects[
        [
            "project_id",
            "rank_score",
            "ds_rank",
            "updated_at",
            "difficulty_score",
            "provider",
        ]
    ].to_dict(orient="records")
    response["ds_metadata"] = "deploy_test"
    response["algo_name"] = "deploy_test"
    response["algo_id"] = "deploy_test"
    response["updated_at"] = str(datetime.strftime(datetime.now(), "%Y-%m-%dT%X%Ez"))
    response["country"] = str(country)
    return response
