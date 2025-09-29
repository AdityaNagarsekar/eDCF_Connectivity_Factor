from django.shortcuts import render

# Create your views here.

from django.http import JsonResponse, HttpResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from .utils.logger import get_logs_as_json, clear_logs
import json
import Driver
from data_structures import *
from algorithms import *
import shutil
import os
from django.conf import settings

allowed_globals = {
    "Circle": Circle,
    "Spiral": Spiral,
    "BarnsleyFern": BarnsleyFern,
    "Mask": Mask,
    "MandelbrotSet": MandelbrotSet,
    "JuliaSet": JuliaSet,
    "SierpinskiTriangle": SierpinskiTriangle,
    "SierpinskiCarpet": SierpinskiCarpet,
    "Sphere": Sphere,
    "SinusoidalCurve": SinusoidalCurve,
    "VShapeGenerator": VShapeGenerator,
    "IrisSetosaData": IrisSetosaData,
    "IrisVersicolorData": IrisVersicolorData,
    "IrisVirginicaData": IrisVirginicaData,
    "AndroidData": AndroidData,
    "Sphere4D": Sphere4D,
    # Add any other classes or functions as needed.
    "KNN": KNN,
    "MLP": MLP,
    "DecisionTree": DecisionTree,
    "SVM": SVM
}

def index(request):
    return render(request, 'DriverFront.html')

@csrf_exempt
def process_data(request):

    if request.method == 'POST':

        data = json.loads(request.body)
        driver = Driver.Driver()

        ctrl_buttons_data = data.get('ctrl_buttons_data', {})

        driver.generate_data_ctrl = ctrl_buttons_data.get('generate_data_ctrl')
        driver.linear_transform_ctrl = ctrl_buttons_data.get('linear_transform_ctrl')
        driver.algorithm_train_ctrl = ctrl_buttons_data.get('train_algo_ctrl')
        driver.calculate_grid_params_ctrl = ctrl_buttons_data.get('calc_grid_params_ctrl')
        driver.compute_grid_ctrl = ctrl_buttons_data.get('compute_grid_ctrl')
        driver.extract_boundary_ctrl = ctrl_buttons_data.get('extract_boundary_ctrl')
        driver.detect_fractal_ctrl = ctrl_buttons_data.get('fractal_dim_ctrl')
        driver.calc_connectivity_ctrl = ctrl_buttons_data.get('connectivity_fac_bound_ctrl')
        driver.data_display_ctrl = ctrl_buttons_data.get('data_display_ctrl')
        driver.grid_display_ctrl = ctrl_buttons_data.get('grid_display_ctrl')
        driver.display_ctrl = ctrl_buttons_data.get('boundary_display_ctrl')
        driver.force_grid_ctrl = ctrl_buttons_data.get('force_grid_ctrl')
        driver.direct_conversion_ctrl = ctrl_buttons_data.get('direct_conv_ctrl')
        driver.dynamic_spacing_ctrl = ctrl_buttons_data.get('dynamic_spacing_ctrl')
        driver.range_analysis_ctrl = ctrl_buttons_data.get('analysis_ctrl')
        driver.range_display_ctrl = ctrl_buttons_data.get('display_analysis_ctrl')
        driver.display_object_grid_ctrl = ctrl_buttons_data.get('display_force_grid_ctrl')
        driver.display_hatch_ctrl = ctrl_buttons_data.get('display_hatch_ctrl')
        driver.hatch_connectivity_ctrl = ctrl_buttons_data.get('connectivity_hatch_ctrl')
        driver.force_grid_connectivity_ctrl = ctrl_buttons_data.get('connectivity_object_ctrl')
        driver.topological_dimension_ctrl = ctrl_buttons_data.get('estimate_topo_ctrl')
        driver.fractal_object_ctrl = ctrl_buttons_data.get('estimate_frac_ctrl')
        driver.interpret_ctrl = ctrl_buttons_data.get('gen_rep_ctrl')
        driver.connectivity_deterioration_ctrl = ctrl_buttons_data.get('connectivity_deterioration_ctrl')
        driver.connectivity_deterioration_display_ctrl = ctrl_buttons_data.get('display_cdet_ctrl')
        driver.save_force_ctrl = ctrl_buttons_data.get('save_force_ctrl')
        driver.connectivity_deterioration_save_ctrl = ctrl_buttons_data.get('save_cdet_ctrl')
        driver.save_ctrl = ctrl_buttons_data.get('save_ctrl')
        driver.time_analysis_ctrl = ctrl_buttons_data.get('time_analysis_ctrl')
        driver.display_time_analysis_ctrl = ctrl_buttons_data.get('display_time_analysis_ctrl')
        driver.save_time_ctrl = ctrl_buttons_data.get('save_time_ctrl')
        driver.clear_ctrl = ctrl_buttons_data.get('clear_ctrl')

        input_data = data.get('input_data')

        try:
            driver.data_objects = eval(input_data.get('data_objects'), allowed_globals)
        except Exception:
            return JsonResponse({"error": "Evalutaion failed"}, status=520)
        
        try:
            driver.num_points = eval(input_data.get('num_points'))
        except Exception:
            return JsonResponse({"error": "Evalutaion failed"}, status=520)
        
        try:
            driver.algorithm = eval(input_data.get('algorithm'), allowed_globals)
        except Exception:
            return JsonResponse({"error": "Evalutaion failed"}, status=520)
        
        try:
            driver.data_objects_names = eval(input_data.get('data_names'))
        except Exception:
            return JsonResponse({"error": "Evalutaion failed"}, status=520)
        
        try:
            driver.data_objects_names = eval(input_data.get('data_names'))
        except Exception:
            return JsonResponse({"error": "Evalutaion failed"}, status=520)
        
        driver.dataset_name = input_data.get('dataset')

        driver.algorithm_name = input_data.get('algorithm_name')

        driver.grid_space = float(input_data.get('grid_spacing'))

        driver.bounds_buffer = float(input_data.get('bounds_buffer'))
        
        driver.grid_divide = int(input_data.get('grid_divisions'))

        driver.neighbour_set_method[0] = input_data.get('neighbour_set_boundary')

        driver.neighbour_set_method[1] = input_data.get('neighbour_set_connectivity')

        driver.directory_name = input_data.get('directory')
        
        try:
            driver.force_identity = eval(input_data.get('force_identity'))
        except Exception:
            return JsonResponse({"error": "Evalutaion failed"}, status=520)

        driver.scale_grid_divide = input_data.get('grid_division_scale')

        driver.scale_grid_space = input_data.get('grid_spacing_scale')

        try:
            driver.range_of_div_add = eval(input_data.get('range_div_add'))
        except Exception:
            return JsonResponse({"error": "Evalutaion failed"}, status=520)

        try:
            driver.range_of_div_multiply = eval(input_data.get('range_div_mul'))
        except Exception:
            return JsonResponse({"error": "Evalutaion failed"}, status=520)

        driver.x_points_num = int(input_data.get('x_coordinates_cd'))

        driver.delete_points_max = int(input_data.get('deletion_cd'))

        driver.probability_deter = float(input_data.get('prob_cd'))

        driver.lower_percent_cap = float(input_data.get('lower_perc'))

        driver.upper_percent_cap = float(input_data.get('upper_perc'))

        driver.step = float(input_data.get('step'))

        driver.display_lower = float(input_data.get('display_lower_based'))

        driver.basis = float(input_data.get('basis'))

        advanced_input_data = data.get('advanced_input_data')

        driver.degree = int(advanced_input_data.get('degree'))

        driver.start_checker = float(advanced_input_data.get('start_checker'))
        
        driver.end_checker = float(advanced_input_data.get('end_checker'))

        driver.binary_iteration_limit = int(advanced_input_data.get('binary_limit'))

        driver.changing_factor = int(advanced_input_data.get('div_fac'))

        driver.bias = float(advanced_input_data.get('bias'))

        try:
            driver.main()
        except Exception:
            return JsonResponse({"error": "Execution failed"}, status=521)

        return JsonResponse({"message": "Data processed successfully"})
    
    else:
        return JsonResponse({"error": "Method Not Allowed"}, status=405)
    
def zip_folder(folder_path):

    shutil.make_archive('Results', 'zip', folder_path)
    return 'Results.zip'  # returns the name of the zipped file

@csrf_exempt
def download_zip(request):

    if request.method == "POST":

        data = json.loads(request.body)

        zip_name = zip_folder(data.get('directory'))

        zip_path = os.path.join(settings.MEDIA_ROOT, zip_name)

        if os.path.exists(zip_path):
            return FileResponse(open(zip_path, 'rb'), as_attachment=True, filename=zip_name)
        else:
            return HttpResponse("File not found", status=404)
    
    else:
        return JsonResponse({"error": "Invalid request method"}, status=400)

@csrf_exempt
def delete_zip(request):
    if request.method == "POST":
        data = json.loads(request.body)
        zip_name = zip_folder(data.get('directory')) 
        zip_path = os.path.join(settings.MEDIA_ROOT, zip_name)

        if os.path.exists(zip_path):
            os.remove(zip_path) 
            shutil.rmtree(data.get('directory'))
            return JsonResponse({"message": "File deleted successfully"}, status=200)
        else:
            return JsonResponse({"error": "File not found"}, status=404)
    
    return JsonResponse({"error": "Invalid request method"}, status=400)