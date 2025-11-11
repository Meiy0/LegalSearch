// script.js
document.addEventListener('DOMContentLoaded', () => {
    const searchForm = document.getElementById('searchForm');
    const resultsContainer = document.getElementById('resultsContainer');
    const pdfFileInput = document.getElementById('pdfFile');
    const queryInput = document.getElementById('queryInput');
    const submitButton = document.getElementById('submitButton');

    searchForm.addEventListener('submit', async function(event) {
        event.preventDefault();

        //Validación básica
        if (!pdfFileInput.files[0] || queryInput.value.trim() === '') {
            resultsContainer.innerHTML = '<p class="error-message">Por favor, seleccione un PDF y escriba una consulta.</p>';
            return;
        }

        resultsContainer.innerHTML = '<p class="loading-message">Procesando PDF y buscando... Esto puede tardar varios segundos (extracción y generación de embeddings).</p>';
        submitButton.disabled = true;

        try {
            //Crear el objeto FormData para enviar archivos junto a texto
            const formData = new FormData();
            formData.append('file', pdfFileInput.files[0]); 
            formData.append('query', queryInput.value); 

            //Petición POST al endpoint /search
            const response = await fetch('/search', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            //Lista de resultados
            if (response.ok && Array.isArray(data)) {
                if (data.length === 0) {
                    resultsContainer.innerHTML = '<p>No se encontraron cláusulas relevantes para su consulta (o la similitud es menor al umbral de 0.1).</p>';
                    return;
                }
                
                let htmlContent = '<h3>Cláusulas más relevantes encontradas:</h3>';
                data.forEach((item, index) => {
                    const score = (item.similitud * 100).toFixed(2);
                    htmlContent += `
                        <div class="result-item">
                            <div class="header">
                                <h4>Resultado #${index + 1}</h4>
                                <p><strong>Similitud:</strong> <span class="score">${score}%</span></p>
                            </div>
                            <pre class="clausula-text">${item.clausula}</pre>
                        </div>
                    `;
                });
                resultsContainer.innerHTML = htmlContent;
                
            } else if (data.error) {
                resultsContainer.innerHTML = `<p class="error-message">❌ Error de la aplicación: ${data.error}</p>`;
            } else {
                 resultsContainer.innerHTML = `<p class="error-message">❌ Error desconocido al procesar la respuesta.</p>`;
            }

        } catch (error) {
            resultsContainer.innerHTML = `<p class="error-message">Error de conexión. Asegúrate de que el servidor Flask esté corriendo en el puerto 5000.</p>`;
        } finally {
            submitButton.disabled = false;
        }
    });
});