# Classificació d'exercicis i tutor adaptatiu

Aquest mòdul afegeix dues funcionalitats al projecte sense modificar el contracte de `predict.py`:

1. Classificar cada exercici per tema i analitzar els errors del model per tema, idioma i nivell.
2. Oferir un tutor que adapta el nivell dels exercicis, dona pistes progressives i guarda el progrés de cada estudiant en SQLite.

El codi utilitza la biblioteca estàndard de Python. L'índex d'embeddings és opcional per al tutor; quan està disponible, serveix per prioritzar exercicis semblants i trobar exemples resolts més senzills.

## 1. Classificar el dataset

Des de la carpeta principal del repositori, executa:

```bash
python3 classify_topics.py --input data/train.jsonl --output data/train_topics.jsonl --review data/topics_to_review_train.csv
python3 classify_topics.py --input data/val.jsonl --output data/val_topics.jsonl --review data/topics_to_review_val.csv
```

El classificador afegeix aquests camps a cada exercici:

- `topic`: categoria matemàtica assignada.
- `topic_confidence`: `high`, `medium` o `low`.
- `topic_source`: `rules` o `manual`.
- `topic_scores`: puntuacions internes de les regles utilitzades.

Categories disponibles: `arithmetic`, `algebra`, `geometry`, `number_theory`, `combinatorics`, `probability`, `statistics`, `measurement`, `logic` i `other`.

**Important:** M3Kang no inclou una etiqueta de tema. La classificació automàtica és una primera aproximació basada en paraules i expressions, no una etiqueta oficial ni un classificador semàntic validat. Revisa els casos de confiança `low` i una mostra dels de confiança `medium` abans d'utilitzar les distribucions a la memòria o a la presentació.

## 2. Corregir etiquetes manualment

Crea `data/topic_overrides.json` amb els identificadors que hagis revisat. Pots fer servir `topic_overrides.example.json` com a plantilla. El format és un objecte JSON:

```json
{
  "id-de-lexercici|eng": "geometry",
  "id-dun-altre-exercici|cat": "combinatorics"
}
```

Les claus poden ser `id|language` (recomanat, perquè un mateix exercici pot tenir diverses traduccions) o només `id`. El valor ha de ser una de les categories disponibles.

Torna a executar la classificació afegint `--overrides`:

```bash
python3 classify_topics.py --input data/train.jsonl --output data/train_topics.jsonl --overrides data/topic_overrides.json --review data/topics_to_review_train.csv
python3 classify_topics.py --input data/val.jsonl --output data/val_topics.jsonl --overrides data/topic_overrides.json --review data/topics_to_review_val.csv
```

Les etiquetes manuals es conserven quan es torna a classificar un fitxer que ja les conté.

## 3. Construir l'índex d'exercicis semblants (opcional, recomanat)

Assegura't que el servei d'embeddings del pas 6/7 està actiu i executa:

```bash
python3 build_index.py --input data/train_topics.jsonl --db data/index.db --only-with-options
```

Només s'indexa el conjunt d'entrenament. No construeixis l'índex a partir de `val_topics.jsonl`: la validació s'ha de mantenir separada.

Si no tens el servei d'embeddings actiu, pots saltar aquest pas. El tutor continuarà funcionant amb selecció per tema i nivell, però no podrà prioritzar veïns semànticament semblants ni mostrar l'exemple resolt recuperat.

## 4. Analitzar els errors de l'avaluació

Primer executa `eval.py`, que és el procés existent del repositori:

```bash
python3 eval.py --limit 50 --name baseline
```

Després localitza el nom exacte del CSV que s'ha creat a `results/` i passa'l a l'anàlisi. Per exemple:

```bash
python3 analyze_errors.py --eval results/eval-20261114-120000-baseline.csv --data data/val_topics.jsonl --output-dir results/analysis-baseline
```

La sortida conté:

- `groups.csv`: encerts, total, accuracy i percentatge d'error per tema, idioma, nivell i disponibilitat de les opcions OCR.
- `mistakes.csv`: fins a 10 respostes incorrectes per inspeccionar manualment.
- `report.md`: resum en Markdown amb les taules per dimensió.

El fitxer d'errors és el punt de partida per revisar els casos concrets a l'explorer. No interpretis una categoria amb pocs exercicis com si fos una estimació precisa: mira sempre el recompte al costat de l'accuracy.

## 5. Executar el tutor adaptatiu

Amb les dades classificades, executa:

```bash
python3 tutor_server.py --input data/train_topics.jsonl --index data/index.db --db results/student_profiles.db --port 8001
```

Obre `http://127.0.0.1:8001` al navegador del mateix dispositiu. Si el navegador és en un altre ordinador de la mateixa xarxa, pots iniciar el servidor amb `--host 0.0.0.0` i obrir `http://IP-DE-LA-PLACA:8001`. Aquest servidor no incorpora autenticació: exposa'l només en una xarxa de confiança i fes servir identificadors pseudònims.

La interfície permet:

- Identificar l'estudiant amb un àlies o codi, no cal fer servir el nom real.
- Escollir idioma, tema i nivell inicial.
- Respondre exercicis de resposta múltiple amb les cinc opcions llegides per OCR.
- Demanar pistes progressives: procediment, exemple més senzill del mateix tema (si hi ha índex i servei d'embeddings) i comprovació del plantejament.
- Veure l'accuracy global i el rendiment per tema.

Durant les primeres cinc respostes, el sistema intenta fer una diagnosi amb temes diferents. Després, prioritza el tema amb pitjor accuracy per a aquell estudiant. Un encert fa pujar un nivell i un error el fa baixar, sempre que hi hagi exercicis disponibles. Evita repetir exercicis ja resolts per aquell identificador d'estudiant. Si no hi ha exercicis del tema o nivell exacte, selecciona l'alternativa disponible més propera.

El perfil es desa a `results/student_profiles.db`, localment al dispositiu. No s'envia a cap servei extern. Per reiniciar els perfils, atura el servidor i elimina aquest fitxer; això esborra l'historial local d'estudiants.

## 6. Proves i comprovacions

Executa les proves unitàries:

```bash
python3 -m unittest discover -s tests -v
```

Prova el servidor amb les dades d'exemple (no calen els fitxers reals de M3Kang):

```bash
python3 tutor_server.py --input data/sample_problems.jsonl --db /tmp/tutor-demo.db --port 8001
```

Abans d'entregar, comprova també el contracte oficial. El tutor no ha de canviar `predict.py`, `eval.py` ni `copilot/pipeline.py`:

```bash
python3 predict.py --input data/sample_problems.jsonl --output results/check.csv
```

Verifica que el CSV tingui una fila per exercici i que cada `answer` sigui una lletra entre A i E. Aquest últim comandament necessita que el servidor del model estigui actiu.

## Fitxers afegits

- `copilot/topics.py`: taxonomia i classificador de primera passada.
- `classify_topics.py`: CLI per etiquetar JSONL i generar la cua de revisió.
- `analyze_errors.py`: unió de l'avaluació amb les etiquetes i resum d'errors.
- `copilot/tutor.py`: motor adaptatiu i persistència SQLite.
- `tutor_server.py`: servidor HTTP local i API del tutor.
- `explorer/tutor.html`: interfície web de l'estudiant.
- `tests/`: proves unitàries.

Les dades de M3Kang continuen dins de `data/`, que ja està exclosa de Git al repositori. No les pugis al repositori públic.
