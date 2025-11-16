// static/clasificador.js

document.addEventListener('DOMContentLoaded', () => {
    
    // --- Selectores del DOM (simplificados) ---
    const searchForm = document.getElementById('searchForm');
    const resultsContainer = document.getElementById('resultsContainer');
    const pdfFileInput = document.getElementById('pdfFile');
    const queryInput = document.getElementById('queryInput');
    const submitButton = document.getElementById('submitButton');
    
    // --- Evento 'submit' (Clasificación) ---
    searchForm.addEventListener('submit', async function(event) {
        event.preventDefault();
        
        if (pdfFileInput.files.length === 0 || queryInput.value.trim() === '') {
             resultsContainer.innerHTML = '<p class="error-message">Por favor, seleccione al menos un PDF y escriba una consulta.</p>';
             return; 
        }

        resultsContainer.innerHTML = '<p class="loading-message">Procesando y clasificando todos los documentos...</p>';
        submitButton.disabled = true;
        
        try {
            const formData = new FormData();
            // ¡Adjuntar TODOS los archivos!
            for (const file of pdfFileInput.files) {
                formData.append('file', file);
            }
            formData.append('query', queryInput.value); 
            
            // ¡Apunta a la API que acabamos de crear!
            const response = await fetch('/api/clasificar', { method: 'POST', body: formData });
            const data = await response.json();

            if (response.ok && Array.isArray(data)) {
                if (data.length === 0) {
                    resultsContainer.innerHTML = '<p>No se encontraron documentos relevantes.</p>';
                    return;
                }
                
                let htmlContent = '<h3>Documentos Clasificados:</h3>';
                // Renderiza la LISTA DE DOCUMENTOS
                data.forEach((item, index) => {
                    const score = (item.similitud * 100).toFixed(2);
                    htmlContent += `
                        <div class="result-item">
                            <div class="header">
                                <h4>${index + 1}. ${item.documento}</h4>
                                <p><strong>Similitud Máxima:</strong> <span class="score">${score}%</span></p>
                            </div>
                        </div>
                    `;
                });
                resultsContainer.innerHTML = htmlContent;

            } else if (data.error) {
                resultsContainer.innerHTML = `<p class="error-message">❌ Error de la aplicación: ${data.error}</p>`;
            } else {
                 resultsContainer.innerHTML = `<p class="error-message">❌ Error desconocido.</p>`;
            }
        } catch (error) {
            resultsContainer.innerHTML = `<p class="error-message">Error de conexión: ${error.message}</p>`;
        } finally {
            submitButton.disabled = false;
        }
    });
});