//Capture the form submission / Capturar el envio del formulario
document.getElementById('strokeForm').addEventListener('submit', async function(e) {
    e.preventDefault();

    //Mostrar el indicador de carga / Show loading indicator
    //Cambia el texto del botón a "Procesando..." / Change the button text to "Processing"
    //Deshabilita el botón para evitar múltiples envíos / Disable the button to prevent multiple submissions
    const submitBtn = this.querySelector('input[type="submit"]');
    const originalText = submitBtn.value;
    submitBtn.value = 'Procesando / Processing';
    submitBtn.disabled = true;

    //Recoger lsod atos del formulario / Collect data from the form
    const formData = new FormData(this);

    try{
        //Enviar los datos al servidor / Send data to the server
        const response = await fetch('/predict/', {
            method: 'POST',
            body: formData
        });

        //Convierte el resultado a un JSON / convert result to JSON
        const result = await response.json();

        // Mostrar resultado
        const resultDiv = document.getElementById('result');
        const riskLevel = document.getElementById('riskLevel');
        const probability = document.getElementById('probability');
        
        resultDiv.style.display = 'block';
        probability.textContent = `${(result.probability * 100).toFixed(2)}%`;
        
        if (result.prediction === 1) {
            resultDiv.className = 'result positive';
            riskLevel.textContent = 'Riesgo Alto de Stroke / High Risk of Stroke';

        } else {
            resultDiv.className = 'result negative';
            riskLevel.textContent = 'Riesgo Bajo de Stroke / Low risk of Stroke';
            
        }
        
        // Desplazarse al resultado
        resultDiv.scrollIntoView({ behavior: 'smooth' });


    }catch(error){
        console.error('Error:', error);
        alert('Error al procesar su solicitud / Error processing your request ');
    }finally {
        // Restaurar el botón
        submitBtn.value = originalText;
        submitBtn.disabled = false;
    }





});