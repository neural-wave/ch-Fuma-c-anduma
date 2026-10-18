# MedAnon: guida tecnica alla codebase

> **Nota (aggiornamento successivo):** questo documento precede le correzioni della revisione indipendente
> (audit senza testo originale, modello richiesto + manifest, ID dettati a voce, limite del 3% sui
> laboratori con chiave segreta, metriche di attacco corrette, verifica dei quasi-identificatori). Per lo
> stato attuale vedere `OVERNIGHT_REPORT.md` e `MODEL_EXPERIMENTS.md`.

_Analisi dello stato osservato il 17 ottobre 2026. Claude Code sta ancora modificando il progetto: numeri e dettagli possono cambiare. I percorsi citati sono relativi a questa directory. Analisi basata su sorgenti, configurazione, test e file di risultati; gli esempi sono inventati._

## 1. Panoramica del progetto

La challenge Bosch chiede di anonimizzare `dialogue` e `note` di incontri medico-paziente, conservando per quanto possibile l'informazione utile a prevedere i codici ICD-10. Lo schema JSONL contiene anche `encounter_id`, `dataset` e `icd10_codes` (`code` e `description`); la soluzione lascia questi campi invariati. La valutazione finale deve funzionare in locale, senza modelli cloud e con massimo 4 GB di VRAM. Le metriche ufficiali, su dati non pubblici, valutano privacy, utilità clinica, originalità e scalabilità ([specifica](../../task/NeuralWaveTask_Anonymization%20of_patient_data_public.pdf)).

MedAnon sceglie una soluzione **ibrida e prevalentemente deterministica**: regole per i formati del corpus, spaCy/Presidio per ulteriori entità, lessici clinici per evitare di cancellare diagnosi e farmaci, e trasformazioni diverse secondo il rischio. L'idea chiave è ragionare sull'intero incontro: un nome riconosciuto nella nota, dove le maiuscole aiutano, può essere ritrovato nel dialogo ASR tutto minuscolo. La trasformazione deve restare coerente tra i due campi. Un modello transformer è disponibile, ma **disabilitato nella configurazione normale**.

I dati forniti hanno 140 incontri di sviluppo (`data/patient_doctor_encounters.jsonl`) e 67 incontri di valutazione locale (`data/eval_patient_doctor_encounters.jsonl`), con un incontro JSON per riga. Il codice assume soprattutto **inglese clinico**: la nota usa le maiuscole, il dialogo ha trascrizioni ASR minuscole con tag come `[doctor]` e `[patient]`. Questo spiega perché una NER fondata soltanto sulle maiuscole non sarebbe sufficiente. L'output conserva i nomi dei campi e i codici/descrizioni ICD; quindi il compito è una **riduzione del rischio di identificazione dentro i testi**, non la rimozione delle diagnosi dal record.

## 2. Architettura della pipeline

```text
JSONL (un incontro per riga)
  → cli.py legge i record e carica la configurazione
  → Anonymizer.detect(record)
      → regole + spaCy/Presidio (+ ML opzionale)
      → filtri di score e MedicalGuard
      → EntityRegistry: unisce menzioni, inferisce ruoli, propaga tra i campi
      → resolve_overlaps: sceglie span non sovrapposti
  → Anonymizer.anonymize_dataset(records)
      → conta combinazioni di quasi identificativi e aumenta la generalizzazione
  → Anonymizer.render(record, detection)
      → Transformer.apply sostituisce o mantiene gli span
      → JSONL con stesso schema + file di audit delle decisioni
```

`Span` è l'unità interna: campo, offset iniziale/finale, categoria, sorgente, score, metadati e ragioni. Gli offset si riferiscono **al testo originale**. `render` costruisce le nuove stringhe in ordine, preservando il resto del record con una copia profonda. Le etichette degli speaker, per esempio `[doctor]`, sono escluse dal rilevamento.

Con `k_anonymity.enabled=true` la CLI carica tutto il dataset e fa due passaggi: rilevamento di tutti gli incontri, poi decisione del livello di generalizzazione e scrittura. Con `--no-k-anonymity` e un solo worker usa invece uno streaming record per record. L'API `Anonymizer.anonymize(record)` lavora su un incontro **senza** l'escalation tra record; per questo una demo singola non mostra esattamente la politica del file finale.

### Sequenza effettiva di `detect`

1. `MedicalGuard.prepare` crea il contesto dalle descrizioni ICD del **record corrente**, salvo `use_icd_context=false`. `note_case_map` raccoglie forme con maiuscola dalla nota.
2. Per ogni campo testuale, nell'ordine configurato (`dialogue`, `note`), `RuleDetector.detect` cerca candidati. `SpacyPresidioDetector.detect` aggiunge entità spaCy e pattern Presidio. `MLDetector.detect` viene invocato solo se abilitato e soltanto sui campi configurati; in modalità `cascade` prima si calcolano le righe incerte con `_uncertain_regions`.
3. Sono rimossi i candidati che intersecano i tag speaker. Ai candidati statistici si applicano soglie `min_score` e `_clean_statistical`; la guardia medica può convertirli in decisioni `keep` anziché in sostituzioni. Anche un `PERSON` statistico composto solo da parole comuni è scartato.
4. `EntityRegistry` riceve gli span `PERSON`, unisce alcune menzioni, determina il ruolo e propone ulteriori occorrenze nei due campi. `_propagate_places_orgs` fa un'analoga ricerca per luoghi e organizzazioni. La guardia medica ricontrolla le menzioni propagate.
5. `resolve_overlaps` restituisce span finali non sovrapposti. `render` usa questi span e le policy per produrre testo e audit. Con la CLI standard, le detection di tutti i record vengono calcolate prima della fase di generalizzazione tra record.

**Dettaglio degli offset.** La ricapitalizzazione è una vista temporanea della stessa lunghezza: l'NER vede, per esempio, un nome con iniziale maiuscola, ma gli indici restano quelli del dialogo originale. L'audit riporta gli offset originali; dopo le sostituzioni non sono più offset dell'output. `resolve_overlaps` sceglie l'etichetta vincente per priorità, poi lunghezza e score, ma amplia lo span all'**unione** dei candidati sovrapposti: evita frammenti non anonimizzati e può anche cancellare testo non identificativo ai bordi.

### Una traccia inventata, dalla detection all'output

```text
dialogue: [doctor] good morning , mr. verdi . your a1c is eight point three .
note:     Mr. Verdi, 67-year-old. HbA1c 8.3%. Diagnosis: type 2 diabetes.
```

- La regola `title` propone uno span `PERSON` per `mr. verdi`; spaCy sulla nota può proporre un secondo span. Il registro li può unire, attribuire il ruolo di paziente dal contesto del dialogo e cercare le altre occorrenze di `Verdi`. Il nome diventa un placeholder di ruolo, mantenendo intatti i tag speaker.
- Le regole di età propongono `AGE` per il solo `67`, con valore 67 in `meta`; `Transformer.apply` genera `[60-69]` nella policy di base.
- `_labs` lega `eight point three` e `8.3` all'analita HbA1c. Se entrambi sono riconosciuti e non vengono sovrascritti da uno span di priorità più alta, il seed basato sul record permette una sostituzione numerica coerente in parole e cifre. La diagnosi rimane come testo e gli `icd10_codes` rimangono invariati. L'esatto nuovo valore dipende dal seed e non va inferito da questa traccia.
- Ogni sostituzione produce una decisione audit con offset sul testo *prima* della sostituzione. Il secondo passaggio può allargare la fascia di età se la firma demografica di quell'incontro è troppo rara nel file.

## 3. Tecnologie e algoritmi utilizzati

| Tecnologia | Ruolo effettivo | Stato |
| --- | --- | --- |
| `re` e regole in `rules.py` | Riconoscono età, date, identificativi, contatti, nomi in contesti conversazionali, luoghi, istituzioni, mestieri e alcuni valori di laboratorio, anche scritti in parole. | Attive sempre. |
| spaCy `en_core_web_lg` | NER su nota e dialogo; il dialogo viene prima ricapitalizzato usando la nota e i lessici, senza cambiare gli offset. | Attivo per default. |
| Presidio Analyzer | Fornisce il motore spaCy e riconoscitori validati per email, telefono, URL, SSN, carte, IBAN ecc. Un'istanza vanilla è usata anche per un audit locale separato. | Attivo per default. |
| Lessici locali | Nomi Census/SSA, frequenze `wordfreq`, luoghi, occupazioni, farmaci FDA, termini/eponimi/vocabolario ICD-10-CM; sono caricati da `src/medanon/resources/`. | Attivi; costruzione con `scripts/build_resources.py` solo in preparazione, con rete. |
| GLiNER, GLiNER2, HF token classification | Ulteriori detector NER locali, su tutto il dialogo (`full`) o sulle righe giudicate incerte (`cascade`). | **Opzionali/sperimentali**: `ml.enabled=false`; pesi da preparare e dipendenze in `requirements-ml.txt`. |
| scikit-learn | TF-IDF + classificatore multilabel per stimare la perdita di utilità ICD-10. | Solo valutazione; non anonimizza. |

Il `name_ratio` in `lexicon.py` confronta una stima della frequenza di un token come nome con la frequenza della stessa parola nell'inglese comune. Serve a trattare diversamente un nome plausibile da parole ambigue come `will`; è una **euristica con costanti fissate nel codice**, non una probabilità calibrata. La funzione `medical_strength` assegna 0/1/2 in base a farmaci, termini clinici, vocabolario ICD e morfologia. I lessici sono statici durante l'inferenza.

**Presidio e spaCy sono usati in due modi diversi.** Nel runtime, `SpacyPresidioDetector` costruisce un `NlpEngineProvider` spaCy e legge direttamente `artifacts.entities` per PERSON/LOCATION/ORGANIZATION/NATIONALITY. Se `presidio_patterns=true`, chiama anche `AnalyzerEngine.analyze` limitandolo alle entità strutturate mappate in `PRESIDIO_PATTERNS`; non usa ogni riconoscitore Presidio come detector generico del testo. Nella valutazione, `ReferenceDetector` crea invece un `AnalyzerEngine()` vanilla, con proprie entità (`PERSON`, `LOCATION`, `DATE_TIME` ecc.). È un controllo distinto, ma con **tecnologia condivisa**; non è un gold standard indipendente.

**Il transformer è un'aggiunta, non il centro della pipeline.** `MLDetector` supporta `gliner`, `gliner2` e `hf`. Per GLiNER e GLiNER2 usa etichette in linguaggio naturale mappate alle categorie MedAnon; per `hf` rimappa etichette come PER/LOC/ORG. Il modello gira localmente, su CPU nella configurazione di classe. In assenza di `allow_download`, imposta `HF_HUB_OFFLINE` e `TRANSFORMERS_OFFLINE`; `prepare-models` è il passo separato che può scaricare i pesi. Il confronto in `scripts/compare_detectors.py` prova sia l'intero dialogo sia la cascade; i risultati sperimentali sono nella sezione 6. Il limite della challenge è **VRAM**, ma i file di risultati qui registrano solo memoria del processo CPU: non dimostrano un picco VRAM specifico.

**Dipendenze e configurazione.** `pyproject.toml` definisce il pacchetto `medanon` e l'entry point CLI. `requirements.txt` installa il runtime più il modello `en_core_web_lg`; `requirements-ml.txt` è opzionale. `load_config` parte sempre da `config.py::DEFAULTS`, poi fonde il file YAML **se specificato**, infine gli override della CLI. `configs/default.yaml` documenta una policy sostanzialmente corrispondente ai default, ma non viene caricato implicitamente. `--no-icd`, `--no-k-anonymity` e `--ml` sono override. Il generatore `scripts/build_resources.py` usa rete e pacchetti supplementari durante la preparazione; l'inferenza legge soltanto i file già distribuiti in `resources/`.

Una sottigliezza di riproducibilità: `config.py` imposta per l'ML opzionale `fields: [dialogue, note]`, mentre `configs/default.yaml` e i profili di `compare_detectors.py` limitano l'ML a `[dialogue]`. Pertanto `medanon anonymize --ml ...` **senza** `--config` non riproduce esattamente il confronto dello script: esegue il detector aggiuntivo anche sulla nota. Va fissata esplicitamente la configurazione quando si comparano tempi, falsi positivi e memoria.

## 4. Componenti principali

| File | Responsabilità e punti d'ingresso |
| --- | --- |
| [`src/medanon/cli.py`](../../src/medanon/cli.py) | Comandi `anonymize`, `evaluate`, `bench`, `canary`, `demo`, `prepare-models`; lettura/scrittura JSONL, audit e worker paralleli. |
| [`src/medanon/config.py`](../../src/medanon/config.py), [`configs/default.yaml`](../../configs/default.yaml) | Valori di default e merge ricorsivo delle opzioni: detector, soglie, policy per categoria, seed, `k=3`. Il file YAML è applicato **solo se passato con `--config`**; i default Python contengono la configurazione normale. |
| [`src/medanon/pipeline.py`](../../src/medanon/pipeline.py) | `Anonymizer.detect`, `render`, `anonymize`, `anonymize_dataset`; coordina i moduli e l'escalation. |
| [`src/medanon/rules.py`](../../src/medanon/rules.py) | `RuleDetector` e pattern specifici del corpus; `LABS` contiene intervalli e soglie usati per i laboratori. |
| [`src/medanon/ner.py`](../../src/medanon/ner.py) | `SpacyPresidioDetector`, `note_case_map`, `truecase`, `MLDetector`; traduce le etichette dei modelli nelle categorie MedAnon. |
| [`src/medanon/lexicon.py`](../../src/medanon/lexicon.py), [`src/medanon/medical.py`](../../src/medanon/medical.py) | Caricano i dizionari; `MedicalGuard.prepare/check` protegge menzioni cliniche scambiate per identificativi. |
| [`src/medanon/entities.py`](../../src/medanon/entities.py) | `EntityRegistry`: raggruppa nomi, vota il ruolo (paziente, clinico, familiare, animale), assegna placeholder e cerca altre occorrenze nei due campi. |
| [`src/medanon/spans.py`](../../src/medanon/spans.py), [`src/medanon/transform.py`](../../src/medanon/transform.py) | Priorità e fusione degli span; `Transformer.apply`, fasce d'età, gerarchia geografica e perturbazione deterministica dei laboratori. |
| [`src/medanon/evaluation/`](../../src/medanon/evaluation/) | Metriche, dati sintetici con etichette note, canary inseriti nei record, attacchi di linkage e diagnostica. |
| [`scripts/run_experiments.py`](../../scripts/run_experiments.py), [`scripts/compare_detectors.py`](../../scripts/compare_detectors.py) | Rigenerano output/metriche e confrontano i detector; il secondo esegue ogni profilo in un processo separato. |

### Oggetti che passano tra i moduli

`Span` (`spans.py`) contiene `field`, `start`, `end`, `label`, `source`, `score`, `text`, `meta`, `reasons`. Le regole salvano, per esempio, l'età numerica o tipo/valore/precisione di un laboratorio in `meta`. Il registro `PersonEntity` (`entities.py`) contiene token e forme del nome, menzioni, voti dei ruoli e placeholder finale. `MedicalGuard.prepare` restituisce un `RecordContext` con token e frasi delle descrizioni ICD. La detection di un incontro è la tupla `(texts, spans, registry, kept)`: `kept` contiene decisioni di mantenimento generate dalla guardia prima del rendering.

La CLI produce un file JSONL anonimizzato e uno `.audit.jsonl` con **una riga per decisione**, non necessariamente una riga per incontro. Il risultato del benchmark completo è `outputs/results.json`; le uscite in `outputs/` sono artefatti generati, non sorgenti. La funzione `integrity` controlla che campi, ICD e metadati restino invariati confrontando `encounter_id` tra input e output; non verifica, da sola, che nel testo non resti alcun identificativo.

## 5. Logica di anonimizzazione

1. **Candidati sensibili.** Le regole sfruttano indizi come titoli, saluti, `date of birth`, MRN, formule di età, preposizioni per luoghi e frasi sul lavoro. spaCy aggiunge PERSON/LOC/ORG/NORP; Presidio aggiunge identificativi strutturati. I detector statistici sono filtrati per score, parole comuni e plausibilità lessicale. La ricapitalizzazione del dialogo usa forme viste nella nota, nomi probabili e località; mantiene la lunghezza della stringa per conservare gli offset.
2. **Protezione medica.** `MedicalGuard` controlla candidati PERSON/ORGANIZATION/LOCATION/NATIONALITY contro eponimi, farmaci, termini clinici e descrizioni ICD del record. Il contesto sintattico forte può prevalere sul dizionario: nell'esempio sintetico, `Mr. Parkinson` è un nome, `Parkinson's disease` è una diagnosi. Con `--no-icd` non usa le descrizioni del singolo record, ma mantiene i lessici globali.
3. **Coerenza.** `EntityRegistry` unisce menzioni che condividono token, inferisce il ruolo da titolo/posizione/speaker e propaga le forme confermate tra `note` e `dialogue`; la propagazione di luoghi e organizzazioni è separata. I casi ambigui richiedono indizi aggiuntivi. Gli span sovrapposti vengono uniti e classificati usando priorità (`DOB` sopra `PERSON`, `PERSON` sopra `LOCATION` ecc.).
4. **Trasformazione.** Nomi → `[PATIENT_NAME]`, `[CLINICIAN_NAME]` ecc.; età → fasce di dieci anni (più specifiche in pediatria); date, DOB, ID, contatti e indirizzi → placeholder; mestieri → categoria; paesi riconosciuti → regione del mondo. L'etnia è mantenuta dalla policy normale. I valori di laboratorio riconosciuti possono essere spostati al massimo del 3%, con seed stabile per incontro/analita/valore, **solo entro le soglie codificate in `LABS`**; se non esiste un valore distinto alla stessa precisione, restano invariati. Non è una garanzia clinica generale: copre solo analiti e soglie elencati nel codice.
5. **Tra record.** Su un file con più incontri, una firma composta da prima età rilevata, sesso inferito dal testo, categorie professionali ed etnie rilevate viene confrontata con le altre. Per gruppi sotto `k=3`, il livello 1 usa `band_width=20` per l'età e sopprime il mestiere; il livello 2 usa età grossolana e sopprime l'etnia. La funzione delle fasce mantiene però intervalli pediatrici specifici e il top-code `90+` anche con larghezza 20. Questo è un tentativo di generalizzazione di alcuni quasi identificativi, **non una prova formale di k-anonimato dell'intero testo**.

### Quali categorie vengono effettivamente trattate

| Categoria | Rilevamento principale | Policy normale |
| --- | --- | --- |
| `PERSON` | Titoli, vocativi, relazioni, nomi dettati, intestazioni; spaCy e ML opzionale. | Placeholder con ruolo inferito. Titoli come `Dr.` possono rientrare nello span sostituito. |
| `DOB`, `DATE`, `YEAR`, `MONTH`, `WEEKDAY`, `HOLIDAY` | Regex di calendario, date pronunciate, cue di nascita, riferimenti temporali. | Placeholder distinto per categoria. Le durate relative possono essere lasciate per utilità clinica. |
| `AGE` | Espressioni `N-year-old`, `age N`, numeri in parole, risposta a «how old». | Solo il numero diventa una fascia; il resto della frase resta. |
| `ID`, `CONTACT`, `ADDRESS` | MRN e altri ID con cue, lunghe cifre, telefono/email/URL/indirizzi e pattern Presidio. | Placeholder. |
| `LOCATION`, `ORGANIZATION` | Gazetteer con indizi di luogo, nomi di ospedali/scuole/datori di lavoro, spaCy e ML opzionale. | Luogo → regione o placeholder; organizzazione → tipo, per esempio struttura sanitaria o datore. |
| `OCCUPATION` | Vocabolario di mestieri solo con indizi professionali vicini. | Categoria ampia, poi eventuale soppressione nel secondo passaggio. |
| `NATIONALITY`, `ETHNICITY` | NER e regole in contesto demografico. | Nazionalità → placeholder; etnia normalmente mantenuta, salvo escalation. |
| `LAB_VALUE` | Analita fra quelli elencati in `LABS`, seguito da un valore numerico o in parole plausibile. | Perturbazione deterministica se c'è un candidato nella stessa categoria clinica; altrimenti valore invariato. |

**Guardia clinica nel dettaglio.** Si applica a PERSON/ORGANIZATION/LOCATION/NATIONALITY, non a ogni categoria. Prima intercetta nomi di sistemi EHR/ASR e intestazioni di sezione. Poi confronta i token con farmaci, termini medici, abbreviazioni, eponimi e parole della descrizione ICD. Un eponimo è difeso soprattutto quando è seguito immediatamente da una testa medica come `disease` o `palsy`; un titolo personale o un contesto sintattico forte può invece imporre la lettura come identificativo. Ciò evita una regola ingenua del tipo «ogni parola in ICD si conserva». Rimangono casi ambigui: i dizionari e le descrizioni non danno certezza semantica, e un termine clinico poco comune o una nuova struttura ospedaliera possono essere classificati male.

**Ruoli e propagazione nel dettaglio.** Un'indicazione esplicita di ruolo (`role_hint`) pesa 3 voti, un titolo senza ruolo certo o un saluto pesa meno; una menzione nella nota dà anche un debole voto a `PATIENT`. `EntityRegistry._find` riusa un'entità quando l'insieme di token di una menzione è sottoinsieme dell'altra o viceversa. L'entità è considerata abbastanza forte per propagare se un indizio è forte, lo score è almeno 0,75 o appare nella nota. La ricerca successiva è case-insensitive, ma nomi anche comuni richiedono titolo, vicinanza a un altro token, vocativo o contesto con maiuscole. Questo migliora il richiamo; due persone che condividono solo un nome o cognome possono però essere fuse, quindi il ruolo del placeholder non è infallibile. `PATIENT_NAME` non riceve suffissi numerici: più persone classificate come paziente possono collassare nello stesso placeholder.

**Laboratori: dettaglio importante sul 3%.** `perturb_lab` genera candidati a passi della precisione riportata (0,1 se il valore ha una cifra decimale), scarta quelli fuori intervallo o oltre una soglia della tabella e sceglie in modo riproducibile con hash del seed, `encounter_id`, analita e valore. Il codice imposta però `max_steps = max(1, round(valore * 0,03 / passo))`: quando il 3% è *meno di un passo*, forza comunque un passo. Verifica sintetica diretta: `TSH 0,1` con una cifra decimale può diventare `0,2`, cioè **+100%**, pur restando nella categoria codificata. Anche `creatinina 0,7 → 0,8` è circa +14,3%. Quindi la frase «variazione ≤3%» presente in commenti/README **non è garantita dall'implementazione**. Il vincolo sulle soglie è anch'esso limitato alla tabella statica `LABS`, non a ogni possibile contesto clinico.

**Generalizzazione tra record: cosa conta davvero.** `quasi_identifiers` prende la prima età rilevata, stima `M/F/U` contando pronomi e parole di genere, usa le categorie di mestieri e le etnie rilevate. L'età è presa prima dalla nota, se presente. A ogni livello l'algoritmo conta le firme correnti e promuove alcune righe con frequenza sotto `k`; poi produce l'output senza un controllo finale delle firme pubblicate. Inoltre la condizione che abilita l'escalation ignora il sesso quando età, lavoro ed etnia mancano. Sono ragioni verificabili nel codice per cui `k=3` è una *euristica di rischio*, non una proprietà dimostrata.

Esempio *inventato*: `La paziente Ada Rossi, 67 anni, lavora come autista` potrebbe diventare `La paziente [PATIENT_NAME], [60-69] anni, lavora come [OCCUPATION: transport]`. L'esempio illustra le policy; il riconoscimento reale dipende dai pattern del corpus, prevalentemente in inglese.

L'audit registra categoria, azione, sostituzione, sorgenti, score e spiegazione. Non è una giustificazione clinica verificata da un esperto: la spiegazione viene composta dalle regole e dai metadati del detector.

Attenzione alla distinzione fra `keep` e sostituzione: per gli span da sostituire `render` non salva `original` per default; per i candidati clinici conservati, `_keep_decision` salva invece sempre `text: s.text`. Il test controlla soltanto l'assenza della **chiave** `original`, quindi non intercetta questo percorso. Nei file di audit già generati, 294 righe sullo split eval e 624 sullo sviluppo hanno una chiave `text` (conteggio delle chiavi, senza riportarne i contenuti). Se il filtro conserva per errore un identificativo, anche l'audit lo replica.

## 6. Valutazione e risultati disponibili

`tests/test_units.py` e `tests/test_pipeline.py` coprono pattern, numeri parlati, eponimi/farmaci, coerenza dei placeholder, soglie di laboratorio, schema JSONL, determinismo, opzione senza ICD e un caso di escalation. Ho eseguito `D:\NEURAL_WAVE\.venv\Scripts\python.exe -m pytest -q`: **49 test passati, 1 warning di deprecazione PyTorch, circa 60 secondi**. La suite non contiene un gold standard annotato su dati reali e non verifica la fuga tramite la chiave `text` dell'audit o il vero limite relativo del 3% sui valori piccoli. I numeri sotto sono quelli già salvati in `outputs/results.json` e possono essere superati mentre Claude lavora.

| Misura locale sullo split eval (67 record) | Valore registrato | Come leggerlo |
| --- | ---: | --- |
| Integrità struttura/codici | 67/67 senza problemi | I campi non testuali e gli ICD sono invariati. |
| Elementi privati assenti secondo detector di riferimento | 87,09% | Presidio individua elementi nell'originale e controlla se la stessa stringa sopravvive; esclude alcune espressioni temporali e cliniche. È un proxy, non recall su annotazioni umane. |
| Nomi-seme sopravvissuti | 0/62 | I semi provengono da pattern ad alta precisione nella nota, quindi non coprono tutti i nomi. |
| Richiamo dei concetti medici lessicali | 99,45% | Misura parole/bigrammi nel lessico del progetto; può non cogliere relazioni, negazioni e dettagli numerici. |
| Termini ICD presenti nel testo e conservati | 100% di 246 | Verifica parole della descrizione ICD, non tutta l'informazione diagnostica. |
| F1 ICD-10 del coder proxy | 0,3836 → 0,3836 | TF-IDF + regressione logistica su 38 codici visti almeno due volte nel training; delta nullo **non** dimostra che ogni diagnosi sia preservata. |
| Tempo medio / memoria massima del processo | 0,328 s per record / circa 2,53 GB | Misure del run locale su CPU; non sono una misura di VRAM né una garanzia su altre macchine. |

Il benchmark sintetico `evaluation/synthetic.py` genera testo con span privati e «trappole» cliniche note: nel run salvato, sullo stile ACI ha **99,59%** di copertura degli span, **95,1%** di precisione sui caratteri sostituiti e **100%** di trappole conservate. Il canary benchmark inserisce identificativi fittizi in contesti reali e registra **98,3%** di copertura complessiva (95,2% per PERSON). Sono test utili, ma gli esempi seguono template del progetto; i canary non etichettano gli identificativi originali presenti nei record.

### Che cosa misurano i proxy

- `synthetic.score`: `recall_coverage` richiede che tutti i caratteri alfanumerici di ogni span privato marcato siano coperti da una decisione di sostituzione; `recall_string` controlla che la stringa originale non sopravviva; `char_precision` misura la quota di caratteri sostituiti inclusi negli span gold; `trap_preservation` misura i termini clinici segnati come trappole non toccati. Il dataset è generato dal progetto stesso.
- `canary.evaluate`: inserisce identificativi sintetici in dialoghi e note reali, poi misura la copertura dei caratteri marcati. Mostra il comportamento nel contesto reale ma **non** trova automaticamente gli identificativi originali del dataset.
- `metrics.leakage_audit`: un Presidio vanilla cerca entità nell'input; per ognuna conta se la *stessa stringa* manca nell'output. Può giudicare rimossa un'informazione che sopravvive in altra forma, e può non vedere entità che Presidio non riconosce. Filtra riferimenti temporali non ancorati al calendario e alcuni termini clinici, quindi il denominatore non comprende tutto il testo.
- `metrics.concept_recall` usa un lessico ricavato in parte dalle stesse risorse che proteggono i termini in `MedicalGuard`; `icd_term_retention` cerca singole parole delle descrizioni ICD. Il loro accordo con la pipeline è utile per regressioni, ma non sostituisce una valutazione semantica indipendente. `icd_proxy_f1` allena un classificatore TF-IDF/unigrammi-bigrammi su 140 incontri e valuta 38 codici con almeno due esempi di training: lo score assoluto 0,3836 è modesto, quindi una variazione zero va letta con cautela.

### Confronto dei detector opzionali

Il file `outputs/detector_comparison.json` contiene ora **sette profili completi**, sul benchmark dello script (60 sintetici ACI, 15 sintetici di dimissione e 60 record di sviluppo con canary). Numeri di questo esperimento, non della submission finale:

| Profilo | Copertura sintetica | Canary complessivo | PERSON nel dialogo con canary | Sostituzioni sintetiche interamente spurie | Secondi per record con canary | RAM picco MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Regole + spaCy (default) | 99,52% | 98,6% | 86,5% | 0 | 0,416 | 1204 |
| + GLiNER full | 99,57% | 99,8% | 98,1% | 30 | 3,726 | 2974 |
| + GLiNER2 full | 99,71% | 100% | 100% | 13 | 5,208 | 2701 |
| + BERT NER full | 99,71% | 100% | 100% | 0 | 2,308 | 1705 |
| + GLiNER cascade | 99,57% | 99,8% | 98,1% | 1 | 1,948 | 3033 |
| + GLiNER2 cascade | 99,71% | 100% | 100% | 0 | 2,935 | 3097 |
| + BERT NER cascade | 99,71% | 100% | 100% | 0 | 1,421 | 1637 |

La differenza più evidente è il richiamo dei **nomi nel dialogo** per i canary, ma il transformer aumenta tempo e RAM. I «0 falsi positivi» sintetici non significano zero errori sul corpus reale: sul reale lo script ha solo un proxy di edit su parole comuni/mediche, senza annotazioni. I tempi sono misurati sul particolare computer usato, e la RAM è il working set del processo; non sono misure di VRAM. Il profilo `hf` usa `dslim/bert-base-NER-uncased` solo in questo confronto, non nel default.

`evaluation/attacks.py` simula tre attacchi: collegamento con età/sesso/lavoro/luogo, collegamento tramite valori esatti di laboratorio e ricerca di nomi residui. Il run eval riporta un tasso di candidati unici di 55,22% sull'originale e 5,97% sull'output; il test di collegamento tramite laboratori trova correttamente e in modo univoco il record nel 76,47% dei 17 target con laboratorio originale e nell'11,76% sull'output. **Il primo dato non è una percentuale validata di persone reidentificate**: `linkage_attack` usa `1/max(1, numero_candidati)` senza verificare che il vero record sia fra i candidati; quindi persino zero candidati contano come probabilità 1. In una diramazione la presenza di una località residua forza una lista con un solo candidato, ancora senza verificare l'identità. Sono problemi dell'attacco simulato, non prova che l'anonimizzazione sia inefficace. Il test dei laboratori, invece, controlla se l'indice del record corretto è fra i candidati. L'attacco sui nomi residui usa nomi-seme ricavati con pattern nella nota, quindi ha una copertura limitata. **Nessuno di questi risultati è il punteggio ufficiale Bosch**.

L'ablation `--no-icd` ha prodotto output identico al profilo normale in 66 incontri eval su 67, con gli stessi proxy aggregati salvati. È un indizio che i lessici globali proteggono molti termini anche senza le descrizioni ICD del record; non dimostra robustezza generale su diagnosi nuove. Non risultano ancora metriche ufficiali del dataset Bosch nascosto.

## 7. Punti di forza, limiti e prossimi passi

**Punti di forza.** La separazione tra rilevamento, guardia clinica e trasformazione rende le decisioni ispezionabili. Il trattamento del dialogo minuscolo e la propagazione fra campi rispondono a problemi specifici del dataset. Il percorso di default usa CPU e lessici locali; `--no-icd`, audit, benchmark sintetici e canary permettono ablation e debugging. Il confronto con ML opzionale è misurato, anziché assunto migliore.

### Problemi osservati nel sorgente

1. **Audit con superfici originali (`pipeline.py::_keep_decision`).** La chiave `text` è scritta per i candidati conservati indipendentemente da `--audit-include-text`; il test guarda solo la chiave `original`. Non è dimostrato che le 294 superfici eval siano identificativi, ma il percorso rende possibile una fuga se la guardia classifica male. Correzione proposta: audit senza superfici per default, test su *tutte* le chiavi testuali dell'audit e audit sensibile separato soltanto su esplicita richiesta.
2. **Vincolo di perturbazione violabile (`transform.py::perturb_lab`).** `max(1, ...)` può produrre uno spostamento molto maggiore del 3% dichiarato. Correzione proposta: rifiutare candidati che superano `abs(nuovo-originale) / abs(originale) <= max_relative`, poi scegliere solo fra quelli che mantengono la categoria. Se la precisione non consente un cambiamento, conservare il valore; aggiungere test per analiti con valori piccoli e per zero.
3. **Metrica di linkage non valida come successo di reidentificazione (`evaluation/attacks.py::linkage_attack`).** Il numero di candidati viene trasformato in «probabilità» senza verificare il vero target; zero candidati è trattato come uno. Correzione proposta: distinguere `target_in_candidates`, `correct_unique`, `wrong_unique`, `no_candidate` e riportare il denominatore. È importante prima di citare il 55,22% → 5,97% davanti ai giudici.
4. **`k=3` non verificato al termine (`pipeline.py::anonymize_dataset`).** La firma contiene pochi attributi, un caso senza età/lavoro/etnia non viene promosso solo in base al sesso, e dopo l'ultimo livello non c'è una verifica della numerosità di ogni gruppo. Correzione proposta: calcolare le firme *dall'output effettivo*, misurare e riportare i gruppi finali `<k`, trattare i record senza dati e aggiungere una politica esplicita quando l'escalation massima non basta. Non chiamare il risultato «k-anonimo» senza delimitare attributi, dataset e garanzie.

### Rischi da misurare, non errori già dimostrati

- **Generalizzazione a nuovi testi.** Regole, liste di nomi e indizi sono tarati sull'inglese ASR/nota; nomi internazionali, istituzioni nuove, abbreviazioni e stili ospedalieri possono sfuggire o creare falsi positivi. Preparare un piccolo gold set annotato per categoria e stile, includendo nomi fuori lessico, e valutare precisione/recall per campo (`note` e `dialogue`) senza esporre i testi nel report.
- **Utilità clinica non completamente misurata.** I proxy lessicali non controllano negazioni, storia familiare, ordine degli eventi, tendenze dei laboratori o specificità dei codici rari. Soglie statiche di `LABS` e fusioni di span possono alterare significato pur conservando parole ICD. Usare un coder indipendente e revisione clinica di un campione di diff; aggiungere test sintetici mirati ai bordi degli span e alle soglie.
- **Memoria e identità.** Con `k` attivo l'intero file e le detection restano in RAM; più worker caricano più copie dei detector. Verificare su un ambiente pulito tempo, RAM e, se si attiva una GPU, VRAM. `encounter_id`, `dataset` e `icd10_codes` sono lasciati invariati per contratto: non bisogna presentare l'output come anonimato medico assoluto né presumere che un ID di un dataset futuro sia sempre non identificativo.
- **Selezione ML.** I sette profili mostrano maggior richiamo di canary per i nomi del dialogo, con costi diversi; mancano annotazioni reali per stimare falsi positivi e veri miss. Il confronto successivo dovrebbe fissare uno split e misurare anche concetti medici e F1 ICD per *ogni* profilo, non scegliere dalla sola recall dei canary.

### Cosa manca per la prossima versione e per la consegna

| Priorità | Lavoro concreto | Verifica di completamento |
| --- | --- | --- |
| 1 | Chiudere le tre anomalie riproducibili: audit, limite relativo dei laboratori, metrica di linkage. | Test mirati nuovi; audit normale senza superfici originali; casi sintetici TSH/creatinina entro limite o invariati; attacco che separa target veri e falsi. |
| 2 | Costruire una valutazione indipendente di privacy e utility con un campione annotato e casi avversari fuori template. | Tabella per etichetta/campo, analisi degli errori, confronto default vs profili ML; nessun dato originale nel report pubblico. |
| 3 | Definire e misurare la garanzia ottenibile per i quasi identificativi dopo il rendering. | Conteggio dei gruppi finali, gestione esplicita dei gruppi `<k`, valutazione del costo in utility. |
| 4 | Consolidare un solo profilo di inferenza riproducibile. | Installazione pulita, modello e lessici disponibili offline, JSONL nello schema richiesto, misura di tempo/RAM e dell'eventuale VRAM. |
| 5 | Chiudere la submission. | `REPORT.md` senza risultati in `TODO`, nomi/ruoli del team nel README, video di massimo due minuti, codice e risorse nella copia/repository finale, commit e prova dell'entry point. |

Al momento dell'ispezione il repository di lavoro su `D:\NEURAL_WAVE\Anonymization_Bosch_NeuralWave` non aveva commit; la copia nel Desktop osservata in precedenza non era allineata. Questi sono aspetti di consegna, distinti dalla qualità dell'algoritmo. I risultati Bosch ufficiali non erano presenti. La prossima decisione progettuale sensata è scegliere una baseline dopo aver corretto misure e audit, poi migliorare il compromesso privacy/utilità con un set annotato che possa smentire le ipotesi attuali.
