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
- Come funzione di minimizzazione dell’errore utilizzo Adam, “adaptive moment estimation” un’estensione della discesa del gradiente stocastica, presentata nel 2015, che unisce i vantaggi di due altre estensioni della discesa del gradiente stocastico ovvero: Adaptive Gradient Algorithm (AdaGrad) che mantiene un tasso di apprendimento per parametro che migliora le prestazioni in caso di problemi con gradienti sparsi e RMSProp, che mantiene anche i tassi di apprendimento che vengono adattati in base alla media delle grandezze recenti dei gradienti per il peso (quanto velocemente sta cambiando). Ciò significa che l’algoritmo funziona bene su problemi online e non stazionari (ad es. Rumoroso). Adam, invece di adattare i tassi di apprendimento dei parametri in base al primo momento medio (la media) come in RMSProp, fa anche uso della media dei secondi momenti dei gradienti (la varianza non centrata).
- Come cella di memoria usiamo una GRU nella quale imagazzino un carattere alla volta con un encoding one-hot, ottengo un alfabeto di ≈ 100 simboli fra maiuscole, minuscole e punteggiatura.
- Il dataset a nostra disposizione è relativamente piccolo, ≈ 5.5M B il che significa che il rischio di overfitting è grande. Per ovviare al problema di memorizzare troppo correttamente la Bibbia e non essere in grado di generare del testo originale utilizziamo un metodo di regolarizzazione, il dropout, che fa fuori il 20% delle cellule ogni ciclo di training.
- Il dropout può essere applicato sia agli input che agli output di un layer denso e questo non fa molta differenza. Osservando la matrice dei pesi di uno strato denso vedo che applicare il dropout agli input equivale a far cadere le linee nella matrice dei pesi mentre applicare il dropout agli output equivale a rimuovere le colonne. Negli RNN è consuetudine aggiungere il dropout agli input in tutti i layer della cella e l’output dell’ultimo layer, che in realtà serve come dropout di input del layer softmax, quindi non è necessario aggiungerlo esplicitamente.
- Il dropout dovrebbe essere applicato agli input e agli output RNN ma non agli stati. In questo approccio, una maschera di esclusione casuale viene ricalcolata ad ogni passo della sequenza srotolata. Questo approccio è chiamato ”naive dropout” ed è quello implementato in questo esperimento.

# Risultati 
- Tensorflow 1.8, Nvidia GeForce 940MX, 384 Cuda Cores. ≈ 24h training
- Si notano già ottimi risultati dopo poche ore (4 o 5) con un hardware casalingo.
- Il dropout fa naturalmente aumentare il tempo di decadimento della funzione di errore ma è il prezzo da pagare per avere una migliore generalizzazione.
- Diverse finestre temporali sono state scelte. Solo con una finestra temporale abbastanza lunga, > 30 la rete è stata in grado di apprendere l’ordine dei versetti e le regole sintattiche come aprire-chiudere virgolette.
- È possibile fare un sampling diverso dei caratteri, scegliendo anche quelli meno probabili con una certa frequenza, che equivale ad alzare o abbassare la “temperatura” del generatore.

# Bibbia 2.0
```
Perché io sono il Signore, tuo Dio, che ha fatto così al mio popolo Israele e ha preso in mondo la tua
parola e io sono il Signore, tuo Dio, e con la sua vita e il suo sangue straniero e tutti gli Israeliti
divorano la tua voce e il tuo popolo Israele si allontana da te che sei scelto.
```
