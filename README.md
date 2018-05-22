# ProfetaVirtuale
A Recurrent Neural Network trained on the Bible, using Karpathy's char-rnn implemented in TensorFlow 

# Introduzione
- Innumerevoli task di apprendimento hanno bisogno di gestire dati sequenziali o serie temporali.
- Le Reti Neurali Ricorrenti (RNN) sono modelli di connessione che catturano la dinamica delle sequenze tramite cicli nella rete dei nodi. A differenza delle reti neurali feed-forward, le reti ricorrenti utilizzano un circuito di feedback collegato alle loro decisioni passate, ingerendo le loro uscite momento dopo momento come input o in alternativa si può dire che utilizzano un vettore di stato che può rappresentare informazione da una finestra di contesto (arbitrariamente lunga).
- Le RNN hanno quindi memoria. L’aggiunta di memoria alle reti neurali ha uno scopo: ci sono informazioni nella sequenza stessa e le reti ricorrenti la usano per eseguire attività che le reti feedforward non possono svolgere.
- È stato dimostrato recentemente che sono un modello Turing-completo, possono eseguire algoritmi arbitrari (1995, Siegelmann).

# Realizzazione
- Idea: Una rete neurale ricorrente allenata sulla Bibbia.
- Come funzione di errore si può usare la Cross Entropy:
- Come funzione di minimizzazione dell’errore utilizzo Adam, “adaptive moment
estimation” un’estensione della discesa del gradiente stocastica, presentata nel
2015, che unisce i vantaggi di due altre estensioni della discesa del gradiente
stocastico ovvero: Adaptive Gradient Algorithm (AdaGrad) che mantiene un
tasso di apprendimento per parametro che migliora le prestazioni in caso di
problemi con gradienti sparsi e RMSProp, che mantiene anche i tassi di
apprendimento che vengono adattati in base alla media delle grandezze recenti
dei gradienti per il peso (quanto velocemente sta cambiando). Ciò significa che
l’algoritmo funziona bene su problemi online e non stazionari (ad es. Rumoroso).
Adam, invece di adattare i tassi di apprendimento dei parametri in base al primo
momento medio (la media) come in RMSProp, fa anche uso della media dei
secondi momenti dei gradienti (la varianza non centrata).
