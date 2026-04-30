class SentimentClassifier {
    constructor() {
        this.model = null;
        this.vocabulary = {};
        this.maxLen = 0;
        this.vocabSize = 0;
        this.lossChart = null;
        this.accuracyChart = null;
        this.isTraining = false;
        this.isTrained = false;

        this.initCharts();
        this.displayDataset();
        this.displayQuickTests();
        this.setupEventListeners();
    }

    initCharts() {
        const chartDefaults = {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                x: { 
                    title: { display: true, text: 'Epoch', color: '#94a3b8' },
                    ticks: { color: '#64748b' },
                    grid: { color: 'rgba(100, 116, 139, 0.1)' }
                },
                y: {
                    ticks: { color: '#64748b' },
                    grid: { color: 'rgba(100, 116, 139, 0.1)' }
                }
            }
        };

        this.lossChart = new Chart(document.getElementById('lossChart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Loss',
                    data: [],
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0
                }]
            },
            options: {
                ...chartDefaults,
                scales: {
                    ...chartDefaults.scales,
                    y: { ...chartDefaults.scales.y, title: { display: true, text: 'Loss', color: '#94a3b8' } }
                }
            }
        });

        this.accuracyChart = new Chart(document.getElementById('accuracyChart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Accuracy',
                    data: [],
                    borderColor: '#22c55e',
                    backgroundColor: 'rgba(34, 197, 94, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0
                }]
            },
            options: {
                ...chartDefaults,
                scales: {
                    ...chartDefaults.scales,
                    y: { ...chartDefaults.scales.y, min: 0, max: 1, title: { display: true, text: 'Accuracy', color: '#94a3b8' } }
                }
            }
        });
    }

    displayDataset() {
        const container = document.getElementById('datasetDisplay');
        container.innerHTML = trainingData.map((item, idx) => `
            <div class="flex items-center gap-2 bg-slate-800 rounded-lg px-3 py-2 text-sm">
                <span class="text-xs text-slate-500 w-6">${idx + 1}</span>
                <span class="flex-1 truncate text-slate-300">${item.text}</span>
                <span class="px-2 py-0.5 rounded-full text-xs font-medium ${item.label === 1 ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'}">
                    ${item.label === 1 ? 'Pos' : 'Neg'}
                </span>
            </div>
        `).join('');

        document.getElementById('sampleCount').textContent = trainingData.length;
    }

    displayQuickTests() {
        const container = document.getElementById('quickTests');
        container.innerHTML = quickTestPhrases.map(phrase => `
            <button onclick="document.getElementById('inputText').value='${phrase.text.replace(/'/g, "\\'")}'; analyzeSentiment();"
                    class="px-3 py-1.5 rounded-full text-xs font-medium border transition-colors
                    ${phrase.type === 'positive' ? 'border-green-600 text-green-400 hover:bg-green-900' : 
                      phrase.type === 'negative' ? 'border-red-600 text-red-400 hover:bg-red-900' : 
                      'border-yellow-600 text-yellow-400 hover:bg-yellow-900'}">
                ${phrase.text}
            </button>
        `).join('');
    }

    setupEventListeners() {
        const epochsSlider = document.getElementById('epochs');
        const unitsSlider = document.getElementById('hiddenUnits');

        epochsSlider.addEventListener('input', (e) => {
            document.getElementById('epochsValue').textContent = e.target.value;
            document.getElementById('totalEpochs').textContent = e.target.value;
        });

        unitsSlider.addEventListener('input', (e) => {
            document.getElementById('unitsValue').textContent = e.target.value;
        });

        document.getElementById('trainBtn').addEventListener('click', () => this.train());
        document.getElementById('analyzeBtn').addEventListener('click', () => analyzeSentiment());

        document.getElementById('inputText').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') analyzeSentiment();
        });
    }

    buildVocabulary() {
        console.log('GentleAI disponible:', typeof GentleAI !== 'undefined');
        console.log('Pre-procesando corpus con GentleAI...');

        const preprocessResult = GentleAI.preprocessPipeline(trainingData, {
            removeStopWords: false,
            minFrequency: 1
        });

        this.vocabulary = preprocessResult.vocabulary;
        this.maxLen = preprocessResult.seqLength;
        this.vocabSize = preprocessResult.vocabSize;
        this.rawSequences = preprocessResult.sequences;
        this.labels = preprocessResult.labels;
        this.tokenizeFn = preprocessResult.tokenize;

        console.log('Vocabulario:', this.vocabSize, 'palabras');
        console.log('Longitud maxima de secuencia:', this.maxLen);
        console.log('Muestras:', this.rawSequences.length);
        console.log('Ejemplo secuencia:', this.rawSequences[0]);

        document.getElementById('vocabSize').textContent = this.vocabSize;
    }

    textToSequence(text) {
        const { sequences } = GentleAI.encodeTexts([text], this.vocabulary, {
            maxLength: this.maxLen,
            padding: 'post',
            truncating: 'post'
        });

        const normalized = GentleAI.normalizeSequences(sequences, this.vocabSize);
        return normalized[0];
    }

    createModel(hiddenUnits) {
        const model = tf.sequential();

        model.add(tf.layers.embedding({
            inputDim: this.vocabSize,
            outputDim: 16,
            inputLength: this.maxLen,
            embeddingsInitializer: 'glorotUniform'
        }));

        model.add(tf.layers.flatten());

        model.add(tf.layers.dense({
            units: hiddenUnits,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));

        model.add(tf.layers.dropout({ rate: 0.3 }));

        model.add(tf.layers.dense({
            units: Math.floor(hiddenUnits / 2),
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));

        model.add(tf.layers.dense({
            units: 1,
            activation: 'sigmoid'
        }));

        return model;
    }

    prepareData() {
        console.log('Creando tensores...');
        console.log('rawSequences length:', this.rawSequences.length);
        console.log('rawSequences[0]:', this.rawSequences[0]);
        console.log('maxLen:', this.maxLen);

        const xs = tf.tensor2d(this.rawSequences, [this.rawSequences.length, this.maxLen], 'int32');
        const ys = tf.tensor2d(this.labels, [this.labels.length, 1]);

        console.log('Tensor xs shape:', xs.shape);
        console.log('Tensor ys shape:', ys.shape);

        return { xs, ys };
    }

    async train() {
        if (this.isTraining) return;

        const trainBtn = document.getElementById('trainBtn');
        trainBtn.disabled = true;
        trainBtn.innerHTML = '<svg class="animate-spin w-5 h-5" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> Entrenando...';

        document.getElementById('statusDot').className = 'pulse-dot w-2 h-2 bg-yellow-400 rounded-full';
        document.getElementById('statusText').textContent = 'Entrenando modelo...';

        if (this.model) {
            this.model.dispose();
        }

        this.lossChart.data.labels = [];
        this.lossChart.data.datasets[0].data = [];
        this.lossChart.update();

        this.accuracyChart.data.labels = [];
        this.accuracyChart.data.datasets[0].data = [];
        this.accuracyChart.update();

        this.buildVocabulary();
        const { xs, ys } = this.prepareData();

        const learningRate = parseFloat(document.getElementById('learningRate').value);
        const epochs = parseInt(document.getElementById('epochs').value);
        const hiddenUnits = parseInt(document.getElementById('hiddenUnits').value);

        console.log('Compilando modelo...');
        this.model = this.createModel(hiddenUnits);
        this.model.compile({
            optimizer: tf.train.adam(learningRate),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        this.model.summary();
        console.log('Iniciando entrenamiento...');

        this.isTraining = true;
        document.getElementById('totalEpochs').textContent = epochs;

        await this.model.fit(xs, ys, {
            epochs: epochs,
            batchSize: 8,
            validationSplit: 0.15,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    const epochNum = epoch + 1;

                    this.lossChart.data.labels.push(epochNum);
                    this.lossChart.data.datasets[0].data.push(logs.loss);
                    this.lossChart.update('none');

                    this.accuracyChart.data.labels.push(epochNum);
                    this.accuracyChart.data.datasets[0].data.push(logs.acc);
                    this.accuracyChart.update('none');

                    document.getElementById('currentEpoch').textContent = epochNum;
                    document.getElementById('trainingProgress').style.width = `${(epochNum / epochs) * 100}%`;

                    document.getElementById('finalLoss').textContent = logs.loss.toFixed(4);
                    document.getElementById('finalAccuracy').textContent = `${(logs.acc * 100).toFixed(1)}%`;
                }
            }
        });

        xs.dispose();
        ys.dispose();

        this.isTraining = false;
        this.isTrained = true;

        trainBtn.disabled = false;
        trainBtn.innerHTML = '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg> Re-Entrenar Modelo';

        document.getElementById('statusDot').className = 'pulse-dot w-2 h-2 bg-green-400 rounded-full';
        document.getElementById('statusText').textContent = 'Modelo entrenado y listo';

        document.getElementById('inputText').disabled = false;
        document.getElementById('analyzeBtn').disabled = false;
    }

    predict(text) {
        if (!this.isTrained) {
            alert('Primero debes entrenar el modelo.');
            return null;
        }

        const { sequences } = GentleAI.encodeTexts([text], this.vocabulary, {
            maxLength: this.maxLen,
            padding: 'post',
            truncating: 'post'
        });

        const tensor = tf.tensor2d(sequences, [1, this.maxLen], 'int32');
        const prediction = this.model.predict(tensor);
        const score = prediction.dataSync()[0];

        tensor.dispose();
        prediction.dispose();

        return score;
    }
}

const classifier = new SentimentClassifier();

function analyzeSentiment() {
    const text = document.getElementById('inputText').value.trim();
    if (!text) return;

    const score = classifier.predict(text);
    if (score === null) return;

    const positivePercent = (score * 100).toFixed(1);
    const negativePercent = ((1 - score) * 100).toFixed(1);

    const resultContainer = document.getElementById('resultContainer');
    const resultCard = document.getElementById('resultCard');
    const resultEmoji = document.getElementById('resultEmoji');
    const resultLabel = document.getElementById('resultLabel');

    document.getElementById('positivePercent').textContent = `${positivePercent}%`;
    document.getElementById('negativePercent').textContent = `${negativePercent}%`;
    document.getElementById('positiveBar').style.width = `${positivePercent}%`;
    document.getElementById('negativeBar').style.width = `${negativePercent}%`;

    resultContainer.classList.remove('hidden');
    resultCard.classList.remove('positive-glow', 'negative-glow', 'neutral-glow');

    const confidence = Math.abs(score - 0.5) * 2;

    if (score > 0.5) {
        resultEmoji.textContent = '😊';
        resultLabel.textContent = 'POSITIVO';
        resultLabel.className = 'text-3xl font-bold text-green-400';
        resultCard.classList.add('positive-glow');
    } else {
        resultEmoji.textContent = '😞';
        resultLabel.textContent = 'NEGATIVO';
        resultLabel.className = 'text-3xl font-bold text-red-400';
        resultCard.classList.add('negative-glow');
    }

    if (confidence < 0.3) {
        resultCard.classList.add('neutral-glow');
        document.getElementById('confidenceNote').textContent = 
            `⚠️ Confianza baja (${(confidence * 100).toFixed(0)}%): El modelo tiene incertidumbre. Esto puede deberse a frases sarcasticas, ambiguas o fuera del dominio de entrenamiento (Unidad 6.4).`;
    } else if (confidence < 0.6) {
        document.getElementById('confidenceNote').textContent = 
            `Confianza moderada: El modelo identifica la tendencia pero con cierto grado de incertidumbre.`;
    } else {
        document.getElementById('confidenceNote').textContent = 
            `Alta confianza: El modelo esta seguro de su prediccion.`;
    }
}
