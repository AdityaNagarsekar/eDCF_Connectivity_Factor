function status_update(button, num){
    let status = JSON.parse(button.value);
    status = !status;
    button.value = JSON.stringify(status);
    localStorage.setItem(`button${num}`, button.value);
}

function initialize_button(button, num, setPrev){
    button.value = setPrev ? localStorage.getItem(`button${num}`) || "false" : "false";
    localStorage.setItem(`button${num}`, button.value);
}

function initialize_all_buttons(setPrev){
    initialize_button(document.getElementById('generate-data'), 1, setPrev);
    initialize_button(document.getElementById('linear-transform'), 2, setPrev);
    initialize_button(document.getElementById('train-algo'), 3, setPrev);
    initialize_button(document.getElementById('calc-grid-params'), 4, setPrev);
    initialize_button(document.getElementById('compute-grid'), 5, setPrev);
    initialize_button(document.getElementById('extract-boundary'), 6, setPrev);
    initialize_button(document.getElementById('fractal-dim'), 7, setPrev);
    initialize_button(document.getElementById('connectivity-fac-bound'), 8, setPrev);
    initialize_button(document.getElementById('data-display'), 9, setPrev);
    initialize_button(document.getElementById('grid-display'), 10, setPrev);
    initialize_button(document.getElementById('boundary-display'), 11, setPrev);
    initialize_button(document.getElementById('force-grid'), 12, setPrev);
    initialize_button(document.getElementById('direct-conv'), 13, setPrev);
    initialize_button(document.getElementById('dynamic-spacing'), 14, setPrev);
    initialize_button(document.getElementById('analysis'), 15, setPrev);
    initialize_button(document.getElementById('display-analysis'), 16, setPrev);
    initialize_button(document.getElementById('display-force-grid'), 17, setPrev);
    initialize_button(document.getElementById('display-hatch'), 18, setPrev);
    initialize_button(document.getElementById('connectivity-hatch'), 19, setPrev);
    initialize_button(document.getElementById('connectivity-object'), 20, setPrev);
    initialize_button(document.getElementById('estimate-topo'), 21, setPrev);
    initialize_button(document.getElementById('gen-rep'), 22, setPrev);
    initialize_button(document.getElementById('connectivity-deterioration'), 23, setPrev);
    initialize_button(document.getElementById('display-cdet'), 24, setPrev);
    initialize_button(document.getElementById('save-force'), 25, setPrev);
    initialize_button(document.getElementById('save-cdet'), 26, setPrev);
    initialize_button(document.getElementById('save'), 27, setPrev);
    initialize_button(document.getElementById('time-analysis'), 28, setPrev);
    initialize_button(document.getElementById('display-time-analysis'), 29, setPrev);
    initialize_button(document.getElementById('save-time'), 30, setPrev);
    initialize_button(document.getElementById('clear'), 31, setPrev);
    initialize_button(document.getElementById('estimate-frac'), 35, setPrev);
}