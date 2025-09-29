function dataCollector() {
    const ctrl_buttons = {
        generate_data_ctrl: JSON.parse(document.getElementById("generate-data").value),

        linear_transform_ctrl: JSON.parse(document.getElementById("linear-transform").value), 

        train_algo_ctrl: JSON.parse(document.getElementById("train-algo").value), 

        calc_grid_params_ctrl: JSON.parse(document.getElementById("calc-grid-params").value), 

        compute_grid_ctrl: JSON.parse(document.getElementById("compute-grid").value), 

        extract_boundary_ctrl: JSON.parse(document.getElementById("extract-boundary").value), 

        fractal_dim_ctrl: JSON.parse(document.getElementById("fractal-dim").value), 

        connectivity_fac_bound_ctrl: JSON.parse(document.getElementById("connectivity-fac-bound").value), 

        data_display_ctrl: JSON.parse(document.getElementById("data-display").value), 

        grid_display_ctrl: JSON.parse(document.getElementById("grid-display").value), 

        boundary_display_ctrl: JSON.parse(document.getElementById("boundary-display").value), 

        force_grid_ctrl: JSON.parse(document.getElementById("force-grid").value), 

        direct_conv_ctrl: JSON.parse(document.getElementById("direct-conv").value), 

        dynamic_spacing_ctrl: JSON.parse(document.getElementById("dynamic-spacing").value), 

        analysis_ctrl: JSON.parse(document.getElementById("analysis").value), 

        display_analysis_ctrl: JSON.parse(document.getElementById("display-analysis").value), 

        display_force_grid_ctrl: JSON.parse(document.getElementById("display-force-grid").value), 

        display_hatch_ctrl: JSON.parse(document.getElementById("display-hatch").value), 

        connectivity_hatch_ctrl: JSON.parse(document.getElementById("connectivity-hatch").value), 

        connectivity_object_ctrl: JSON.parse(document.getElementById("connectivity-object").value), 
        
        estimate_topo_ctrl: JSON.parse(document.getElementById("estimate-topo").value),

        estimate_frac_ctrl: JSON.parse(document.getElementById("estimate-frac").value),

        gen_rep_ctrl: JSON.parse(document.getElementById("gen-rep").value), 

        connectivity_deterioration_ctrl: JSON.parse(document.getElementById("connectivity-deterioration").value), 

        display_cdet_ctrl: JSON.parse(document.getElementById("display-cdet").value), 

        save_force_ctrl: JSON.parse(document.getElementById("save-force").value), 

        save_cdet_ctrl: JSON.parse(document.getElementById("save-cdet").value), 

        save_ctrl: JSON.parse(document.getElementById("save").value), 

        time_analysis_ctrl: JSON.parse(document.getElementById("time-analysis").value), 

        display_time_analysis_ctrl: JSON.parse(document.getElementById("display-time-analysis").value), 

        save_time_ctrl: JSON.parse(document.getElementById("save-time").value), 
        
        clear_ctrl: JSON.parse(document.getElementById("clear").value)
    };

    const inputs = {
        data_objects: document.getElementById("data-objects").value, 

        num_points: document.getElementById("num-points").value, 

        algorithm: document.getElementById("algorithm").value, 

        data_names: document.getElementById("data-names").value, 

        dataset: document.getElementById("dataset").value, 

        algorithm_name: document.getElementById("algorithm-name").value, 

        grid_spacing: Number(document.getElementById("grid-spacing").value), 

        bounds_buffer: Number(document.getElementById("bounds-buffer").value), 

        grid_divisions: Number(document.getElementById("grid-divisions").value), 

        neighbour_set_boundary: document.getElementById("neighbour-set-boundary").checked, 

        neighbour_set_connectivity: document.getElementById("neighbour-set-connectivity").checked, 

        directory: document.getElementById("directory").value, 

        force_identity: document.getElementById("force-identity").value, 

        grid_spacing_scale: document.getElementById("grid-spacing-scale").value, 

        grid_division_scale: document.getElementById("grid-division-scale").value, 

        range_div_mul: document.getElementById("range-div-mul").value, 

        range_div_add: document.getElementById("range-div-add").value, 

        x_coordinates_cd: Number(document.getElementById("x-coordinates-cd").value), 

        deletion_cd: Number(document.getElementById("deletion-cd").value), 

        prob_cd: Number(document.getElementById("prob-cd").value), 

        lower_perc: Number(document.getElementById("lower-perc").value), 

        upper_perc: Number(document.getElementById("upper-perc").value), 

        step: Number(document.getElementById("step").value), 

        display_lower_based: Number(document.getElementById("display-lower-based").value), 

        basis: Number(document.getElementById("basis").value),
    };

    const advanced_input = {
        degree: Number(document.getElementById("degree").value), 

        start_checker: Number(document.getElementById("start-checker").value), 

        end_checker: Number(document.getElementById("end-checker").value), 

        binary_limit: Number(document.getElementById("binary-limit").value), 

        div_fac: Number(document.getElementById("div-fac").value),

        bias: Number(document.getElementById("bias").value),
    };

    const dataToSend = {
        ctrl_buttons_data: ctrl_buttons, 
        input_data: inputs, 
        advanced_input_data: advanced_input
    };

    return JSON.stringify(dataToSend);
}

function sendData(json_data){
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();

        xhr.open("POST", "/process_data/");
        xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");

        xhr.onload = () => {
            if(xhr.status >= 200 && xhr.status < 400){
                try{
                    const responseData = JSON.parse(xhr.responseText);
                    resolve(responseData);
                }
                catch(error){
                    reject(new Error("Failed to parse JSON response: " + error.message));
                }
            }
            else{
                reject(new Error(`Request failed with status ${xhr.status}: ${xhr.statusText}`));
            }
        };

        xhr.onerror = () => {
            reject(new Error("Network error occured."));
        };

        xhr.send(json_data);
    });
}

const log = {

    value: "", 
    
    get(){
        return document.getElementById('log').innerHTML;
    },

    scrollBottom(){
        document.getElementById('log').scrollTop = document.getElementById('log').scrollHeight;
    }, 
    
    set(content){
        document.getElementById('log').innerHTML = content;
        this.scrollBottom();
    }, 

    update(content){
        this.set(this.get() + `\n${content}`);
        this.value = this.get();
    }
}

function run(button, num){

    if(JSON.parse(button.value) && button.className === "submit"){

        log.value = "Starting Process...\n";

        log.set(log.value);

        log.update("Starting Data Collection...\n");

        json_data = dataCollector();

        log.update("Data Collection Finished.\n");

        log.update("Sending Data Processing Request...\n");

        sendData(json_data)
            .then(data => {
                console.log(`Server Response: ${data.message}`);
                status_update(button, num);
                log.update("Data Processing Successful.\n");
                log.update("Download your data from Download Folder Button Now if Saved or Continue Operation to Download Later.\n");
            })
            .catch(error => {
                console.error(`Error: ${error}`);
                status_update(button, num);
                log.update("Data Processing Unsuccessful.\n");
            });
    }
}

function download(button, num){

    log.value = "Downloading Results\n";

    log.set(log.value);

    if(JSON.parse(button.value) && button.className === "Download-Result-Folder"){
        const data = {
            directory: document.getElementById("directory").value, 
        }
        
        const xhr = new XMLHttpRequest();

        xhr.open("POST", "download_zip/", true);
        xhr.setRequestHeader("Content-Type", "application/json");
        xhr.responseType = "blob";

        xhr.onload = () => {
            if (xhr.status === 200) {
                var blob = xhr.response;

                if (!(blob instanceof Blob)) {
                    console.error("Response is not a Blob:", blob);
                    log.update("Download Failed.\n");
                    return;
                }

                var link = document.createElement("a");
                link.href = window.URL.createObjectURL(blob);
                link.download = "Result.zip";  // Name of the downloaded file
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link); // Clean up
                log.update("Download Successful.\n");

                deleteDownloadedFile(data.directory);

            } else {
                console.error("Download failed", xhr.status);
                log.update("Download Failed.\n");
            }
        };

        xhr.send(JSON.stringify(data));

        status_update(button, num);


    }
}

function deleteDownloadedFile(directory) {
    fetch("delete_zip/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ directory: directory })
    })
    .then(response => response.json())
    .then(data => console.log("File deleted:", data))
    .catch(error => console.error("File deletion failed:", error));
}