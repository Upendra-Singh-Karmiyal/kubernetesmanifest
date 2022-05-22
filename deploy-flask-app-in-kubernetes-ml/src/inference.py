from import_lib_helper import *
from inference_helper import *

app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello World!"


@app.route("/inference", methods=["POST"])
def inference():

    country = request.json["country"]

    ################## read Assest ############################
    model_path, feature_path, inventory_path = get_path(country)
    model = read_pickle(model_path)
    features = read_json(feature_path)
    inventory = read_pickle(inventory_path)
    proj_inv = inventory["project_inventory"]
    pred_feat = proj_inv[features["actv"]]
    ###########################################################

    ######### prediction, rank gen & final response ###########
    rank_score = gen_prediction(model, pred_feat)
    proj_inv["rank_score"] = rank_score
    ranked_projects = projects_rank_gen(proj_inv)
    final_response = create_final_response(country, ranked_projects)
    print("API request suucessful!!")
    ############################################################

    return final_response


if __name__ == "__main__":
    app.run()
