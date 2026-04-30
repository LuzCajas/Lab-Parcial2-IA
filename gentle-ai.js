/**
 * Gentle-AI: Natural Language Processing Module
 * Pre-procesamiento de texto para clasificacion de sentimientos
 * Basado en TensorFlow.js
 */
const GentleAI = (() => {

    const stopWords = new Set([
        'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
        'de', 'del', 'al', 'en', 'con', 'sin', 'por', 'para',
        'que', 'se', 'es', 'son', 'fue', 'ser', 'estar', 'esta',
        'muy', 'mas', 'tan', 'tanto', 'todo', 'todos', 'toda',
        'este', 'esta', 'estos', 'estas', 'ese', 'esa', 'esos',
        'su', 'sus', 'mi', 'tu', 'nos', 'me', 'te', 'le',
        'y', 'o', 'pero', 'si', 'no', 'como', 'cuando',
        'donde', 'quien', 'cual', 'cuales', 'cuanto',
        'ha', 'han', 'he', 'has', 'hay', 'habia',
        'tiene', 'tienen', 'tengo', 'hacer', 'puede',
        'a', 'e', 'o', 'u', 'da', 'dan', 'lo',
        'se', 'le', 'les', 'lo', 'la', 'las', 'los',
        'muy', 'mas', 'ya', 'aun', 'solo', 'solamente',
        'tambien', 'poco', 'mucho', 'bastante', 'algo',
        'nada', 'nadie', 'algun', 'ningun', 'cada',
        'cualquier', 'cualquiera', 'otro', 'otra', 'otros'
    ]);

    const accentMap = {
        'a': /[áàâäã]/g, 'e': /[éèêë]/g, 'i': /[íìîï]/g,
        'o': /[óòôöõ]/g, 'u': /[úùûü]/g, 'n': /[ñ]/g,
        'A': /[ÁÀÂÄÃ]/g, 'E': /[ÉÈÊË]/g, 'I': /[ÍÌÎÏ]/g,
        'O': /[ÓÒÔÖÕ]/g, 'U': /[ÚÙÛÜ]/g, 'N': /[Ñ]/g
    };

    function normalizeText(text) {
        let normalized = text.toLowerCase().trim();
        normalized = normalized.normalize('NFD').replace(/[\u0300-\u036f]/g, '');
        normalized = normalized.replace(/[^\w\s]/g, '');
        normalized = normalized.replace(/\s+/g, ' ');
        return normalized;
    }

    function tokenize(text, options = {}) {
        const { removeStopWords = false, minTokenLength = 2 } = options;
        
        const normalized = normalizeText(text);
        let tokens = normalized.split(' ').filter(t => t.length >= minTokenLength);
        
        if (removeStopWords) {
            tokens = tokens.filter(t => !stopWords.has(t));
        }
        
        return tokens;
    }

    function buildVocabulary(corpus, options = {}) {
        const { removeStopWords = false, minFrequency = 1 } = options;
        
        const wordCounts = {};
        
        corpus.forEach(item => {
            const tokens = tokenize(item.text, { removeStopWords });
            tokens.forEach(token => {
                wordCounts[token] = (wordCounts[token] || 0) + 1;
            });
        });

        const vocabulary = { '<PAD>': 0, '<UNK>': 1 };
        let idx = 2;
        
        const sortedWords = Object.entries(wordCounts)
            .sort((a, b) => b[1] - a[1]);
        
        sortedWords.forEach(([word, count]) => {
            if (count >= minFrequency && !vocabulary[word]) {
                vocabulary[word] = idx++;
            }
        });
        
        return vocabulary;
    }

    function encodeTexts(texts, vocabulary, options = {}) {
        const { maxLength = null, padding = 'post', truncating = 'post' } = options;
        
        const tokenizedTexts = texts.map(text => tokenize(text));
        
        const seqLength = maxLength || Math.max(...tokenizedTexts.map(t => t.length));
        
        const sequences = tokenizedTexts.map(tokens => {
            let sequence = tokens.map(token => vocabulary[token] !== undefined ? vocabulary[token] : 1);
            
            if (sequence.length > seqLength) {
                if (truncating === 'post') {
                    sequence = sequence.slice(0, seqLength);
                } else {
                    sequence = sequence.slice(sequence.length - seqLength);
                }
            }
            
            while (sequence.length < seqLength) {
                if (padding === 'post') {
                    sequence.push(0);
                } else {
                    sequence.unshift(0);
                }
            }
            
            return sequence;
        });
        
        return { sequences, seqLength };
    }

    function normalizeSequences(sequences, vocabSize) {
        return sequences.map(seq => 
            seq.map(token => token / Math.max(vocabSize - 1, 1))
        );
    }

    function preprocessPipeline(corpus, options = {}) {
        const {
            removeStopWords = false,
            minFrequency = 1,
            maxLength = null,
            padding = 'post',
            truncating = 'post'
        } = options;

        const vocabulary = buildVocabulary(corpus, { removeStopWords, minFrequency });
        const texts = corpus.map(item => item.text);
        const { sequences, seqLength } = encodeTexts(texts, vocabulary, { maxLength, padding, truncating });
        const labels = corpus.map(item => item.label);

        return {
            vocabulary,
            sequences,
            labels,
            seqLength,
            vocabSize: Object.keys(vocabulary).length,
            tokenize: (text) => tokenize(text, { removeStopWords })
        };
    }

    function getStopWords() {
        return Array.from(stopWords);
    }

    function addStopWords(words) {
        words.forEach(w => stopWords.add(w.toLowerCase()));
    }

    return {
        tokenize,
        normalizeText,
        buildVocabulary,
        encodeTexts,
        normalizeSequences,
        preprocessPipeline,
        getStopWords,
        addStopWords,
        stopWords
    };
})();
