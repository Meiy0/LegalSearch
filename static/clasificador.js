document.addEventListener('DOMContentLoaded', () => {

    const searchForm = document.getElementById('searchForm');
    const resultsContainer = document.getElementById('resultsContainer');
    const pdfFileInput = document.getElementById('pdfFile');
    const queryInput = document.getElementById('queryInput');
    const submitButton = document.getElementById('submitButton');


    function interpretarResultado(item) {

        if (item.coincidencia_exacta) {
            return "La frase aparece literalmente en el documento.";
        }

        if (item.fuzzy && item.score >= 85) {
            return "El documento contiene una coincidencia muy alta con la frase buscada.";
        }

        if (item.fuzzy && item.score >= 70) {
            return "El documento contiene coincidencias moderadas con la frase buscada.";
        }

        return "No se encontraron coincidencias relevantes con la frase buscada.";
    }


    searchForm.addEventListener('submit', async function(event) {
        event.preventDefault();

        if (pdfFileInput.files.length === 0 || queryInput.value.trim() === '') {
            resultsContainer.innerHTML = '<p class="error-message">Por favor, seleccione PDFs y escriba una palabra.</p>';
            return;
        }

        resultsContainer.innerHTML = '<p class="loading-message">Buscando coincidencias...</p>';
        submitButton.disabled = true;

        try {
            const formData = new FormData();
            for (const file of pdfFileInput.files) {
                formData.append('file', file);
            }
            formData.append('query', queryInput.value);

            const response = await fetch('/api/clasificar', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                resultsContainer.innerHTML = `<p class="error-message">${data.error}</p>`;
                return;
            }

            let html = `<h3>Resultados:</h3>`;

            data.forEach((item, index) => {

                let nivel = "";
                let color = "";

                if (item.coincidencia_exacta) {
                    nivel = "Coincidencia Exacta";
                    color = "high-score";
                } 
                else if (item.score >= 85) {
                    nivel = "Alta relevancia";
                    color = "high-score";
                } 
                else if (item.score >= 70) {
                    nivel = "Relevancia media";
                    color = "medium-score";
                } 
                else {
                    nivel = "Baja relevancia";
                    color = "low-score";
                }

                html += `
                    <div class="result-card ${color}">
                        <h4>${index + 1}. ${item.documento}</h4>

                        <p><strong>${nivel}</strong> — Nivel de coincidencia: ${item.score.toFixed(0)}%</p>

                        <details>
                            <summary>Ver detalles</summary>
                                <ul>
                                    <li><strong>Interpretación:</strong> ${interpretarResultado(item)}</li>
                                    <li><strong>Justificación del sistema:</strong>
                                        El documento presenta un ${item.score.toFixed(0)}% de similitud con respecto a la frase consultada.
                                    </li>
                                </ul>
                        </details>
                    </div>
                `;
            });

            resultsContainer.innerHTML = html;

        } catch (err) {
            resultsContainer.innerHTML = `<p class="error-message">Error de conexión: ${err.message}</p>`;
        } finally {
            submitButton.disabled = false;
        }
    });
});
