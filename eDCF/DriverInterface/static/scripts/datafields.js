function initialize_text_datafield(field, num, setPrev){
    field.value = setPrev ? localStorage.getItem(`field${num}`) || "" : "";
    localStorage.setItem(`field${num}`, field.value);
}

function initialize_number_datafield(field, num, setPrev){
    const checker = setPrev ? localStorage.getItem(`field${num}`) || '' : '';
    field.value = checker ? JSON.parse(checker) : '';
    localStorage.setItem(`field${num}`, JSON.stringify(field.value));
}

function initialize_checkbox_datafield(field, num, setPrev){
    field.checked = JSON.parse(setPrev ? localStorage.getItem(`field${num}`) || 'false' : 'false');
    localStorage.setItem(`field${num}`, JSON.stringify(field.checked));
}

function initialize_all_datafields(setPrev){
    initialize_text_datafield(document.getElementById('data-objects'), 1, setPrev);
    initialize_text_datafield(document.getElementById('num-points'), 2, setPrev);
    initialize_text_datafield(document.getElementById('algorithm'), 3, setPrev);
    initialize_text_datafield(document.getElementById('data-names'), 4, setPrev);
    initialize_text_datafield(document.getElementById('dataset'), 5, setPrev);
    initialize_text_datafield(document.getElementById('algorithm-name'), 6, setPrev);

    initialize_number_datafield(document.getElementById('grid-spacing'), 7, setPrev);
    initialize_number_datafield(document.getElementById('bounds-buffer'), 8, setPrev);
    initialize_number_datafield(document.getElementById('grid-divisions'), 9, setPrev);

    initialize_checkbox_datafield(document.getElementById('neighbour-set-boundary'), 10, setPrev);
    initialize_checkbox_datafield(document.getElementById('neighbour-set-connectivity'), 11, setPrev);

    initialize_text_datafield(document.getElementById('directory'), 12, setPrev);
    initialize_text_datafield(document.getElementById('force-identity'), 13, setPrev);
    initialize_text_datafield(document.getElementById('grid-spacing-scale'), 14, setPrev);
    initialize_text_datafield(document.getElementById('grid-division-scale'), 15, setPrev);
    initialize_text_datafield(document.getElementById('range-div-mul'), 16, setPrev);
    initialize_text_datafield(document.getElementById('range-div-add'), 17, setPrev);

    initialize_number_datafield(document.getElementById('x-coordinates-cd'), 18, setPrev);
    initialize_number_datafield(document.getElementById('deletion-cd'), 19, setPrev);
    initialize_number_datafield(document.getElementById('prob-cd'), 20, setPrev);
    initialize_number_datafield(document.getElementById('lower-perc'), 21, setPrev);
    initialize_number_datafield(document.getElementById('upper-perc'), 22, setPrev);
    initialize_number_datafield(document.getElementById('step'), 23, setPrev);
    initialize_number_datafield(document.getElementById('display-lower-based'), 24, setPrev);
    initialize_number_datafield(document.getElementById('basis'), 25, setPrev);
}