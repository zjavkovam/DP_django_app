from django.shortcuts import render
from django.http import JsonResponse
import joblib
import os
from . import utils
from django.utils.timezone import now


def home(request):
    return render(request, "home.html")

def predict(request):
    if request.method == "POST":
        smiles_or_cas = request.POST.get("inputData", "")
        selected_model = request.POST.get("model", "")
        print("selected model",selected_model)
        
        try:
            # Convert input to features
            processed_input = utils.process_input(smiles_or_cas)
            features = processed_input[1]
            smiles = processed_input[0]

            img_url = utils.get_molecule_img(smiles)

            # Make prediction
            model_result = utils.make_prediction(features, selected_model)
            info = utils.get_chemical_info(smiles)

            # Prepare response
            response = {
                "prediction": "Irritant" if model_result["prediction"]== 1 else "Non-Irritant",
                "timestamp": now().timestamp(),
                "molecule_image": img_url,
                "model_used": selected_model, 
                "roc_curve": model_result["roc_curve"], 
                "trust_score":  model_result["trust_score"],
                "name": info["Chemical Name"],
                "casnr": info["CAS Number"],
                "formula": info["Molecular Formula"],
                "smiles": smiles,
                "shap_values": model_result["shap_plot"],
                "model_parameters": model_result["model_parameters"]
            }
            return JsonResponse(response)

        except ValueError as e:
            # If ValueError is raised, return error message
            error_message = str(e)
            return JsonResponse({'error': error_message}, status=400)

    return render(request, "home.html")

def info(request):
    print("")