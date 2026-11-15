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

**Important:** M3Kang no inclou una etiqueta de tema. La classificació automàtica és una primera aproximació basada en paraules i expressions, no una etiqueta oficial ni un classificador semàntic validat. Les regles distingeixen ara expressions com «perfect square/cube» (teoria de nombres) de les figures geomètriques. Revisa els casos de confiança `low` i una mostra dels de confiança `medium` abans d'utilitzar les distribucions a la memòria o a la presentació.

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

Obre `http://127.0.0.1:8001` al navegador del mateix dispositiu. Si el navegador és en un altre ordinador de la mateixa xarxa, pots iniciar el servidor amb `--host 0.0.0.0` i obrir `http://IP-DE-LA-PLACA:8001`. Aquest servidor no incorpora autenticació: exposa'l només en una xarxa de confiança i fes servir identificadors pseudònims. Per defecte, també busca les figures a `data/images/`; si les tens en una altra carpeta, indica-la amb `--images RUTA`.

La interfície permet:

- Identificar l'estudiant amb un àlies o codi, no cal fer servir el nom real. No hi ha cap identificador compartit per defecte: cada estudiant ha d'utilitzar el seu propi codi per evitar barrejar els historials.
- Veure la imatge original de l'exercici quan existeix a `data/images/`. El servidor detecta el format real del fitxer, perquè alguns recursos tenen extensió `.png` però contingut JPEG.
- Escollir l'idioma i, opcionalment, un tema. **No hi ha selector manual de nivell**: el nivell inicial es calcula automàticament a partir de la mediana dels nivells disponibles per a l'idioma i el tema triats. Això evita demanar a l'estudiant una configuració que el tutor ja adapta durant la sessió.
- Respondre exercicis de resposta múltiple amb les cinc opcions llegides per OCR.
- Demanar fins a tres pistes progressives, generades **una a una** per al problema actiu. Cada pista inclou una explicació centrada en les dades i les condicions reals de l'enunciat, una representació visual en text (esquema, taula, relació o arbre) i una acció concreta per provar.
- Les pistes no reben l'etiqueta de la resposta correcta ni el text de l'opció correcta. La interfície conserva les pistes anteriors perquè l'estudiant pugui seguir-ne la progressió.
- Si el model local no respon, el tutor no deixa la petició penjada: mostra una pista de suport construïda a partir de la pregunta concreta, els nombres detectats i el tema classificat. S'identifica visualment com a pista de suport, no com a resposta generada pel model.
- Després de la tercera pista, el servidor rebutja noves peticions per a aquell exercici. La interfície mostra quantes pistes queden i evita dobles clics mentre el model genera la resposta.
- Veure l'accuracy global i el rendiment per tema.

Durant les primeres cinc respostes, el sistema intenta fer una diagnosi amb temes diferents. Després, prioritza el tema amb pitjor rendiment observat. Un encert fa pujar un nivell i un error el fa baixar, sempre que hi hagi exercicis disponibles. Evita repetir exercicis ja resolts per aquell identificador d'estudiant. Si no hi ha exercicis del tema o nivell exacte, selecciona l'alternativa disponible més propera.

El perfil es desa a `results/student_profiles.db`, localment al dispositiu. No s'envia a cap servei extern. Per reiniciar els perfils, atura el servidor i elimina aquest fitxer; això esborra l'historial local d'estudiants.

## 6. Investigar l'accuracy amb les imatges dels exercicis

L'avaluació textual actual no envia les imatges al model. Això és rellevant perquè una part dels exercicis depèn d'una figura i, en alguns casos, les opcions també són gràfiques. Ara `eval.py` i `predict.py` tenen una opció multimodal explícita; el mode habitual continua sent textual i no canvia el contracte d'`A` a `E`.

Primer, comprova que les imatges corresponents existeixen a `data/images/` i que el servidor de visió amb `--mmproj` està actiu al port 8083 (consulta l'apartat de visió opcional del `README.md`). Després, des de la carpeta principal:

```bash
python3 eval.py --limit 50 --vision --images data/images --name vision50
```

La comparació ha de fer-se amb la mateixa mostra de 50 exercicis. L'avaluació manté el límit oficial de 60 segons per problema; si el servidor visual no hi és o supera el límit, el pipeline retorna la resposta de reserva. Revisa `image_sent`, `status`, la latència i l'accuracy abans d'interpretar el resultat. La visió és experimental i pot requerir memòria addicional a la placa.

Per utilitzar la mateixa opció en una execució de predicció:

```bash
python3 predict.py --input data/val.jsonl --output results/predictions-vision.csv --vision --images data/images
```

No consideris que la visió millora el model fins que l'experiment ho demostri. Després d'una prova de 50 exercicis, confirma qualsevol millora amb tot el conjunt de validació (`--limit 0`).

## 7. Proves i comprovacions

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

## Fitxers implicats

- `copilot/topics.py`: taxonomia i classificador de primera passada.
- `classify_topics.py`: CLI per etiquetar JSONL i generar la cua de revisió.
- `analyze_errors.py`: unió de l'avaluació amb les etiquetes i resum d'errors.
- `copilot/tutor.py`: motor adaptatiu, nivell inicial automàtic, pistes progressives i persistència SQLite.
- `tutor_server.py`: servidor HTTP local i API del tutor.
- `explorer/tutor.html`: interfície web de l'estudiant, amb imatge original i pistes visuals acumulatives.
- `eval.py` i `predict.py`: suport opcional per enviar les imatges a un servidor multimodal local.
- `tests/`: proves unitàries del motor i de les pistes.

Les dades de M3Kang continuen dins de `data/`, que ja està exclosa de Git al repositori. No les pugis al repositori públic.
