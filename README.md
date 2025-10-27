# TrackingEvaluation
Evaluate tracking

Código do TrackEval - https://github.com/JonathonLuiten/TrackEval

Crie um ambiente virtual

`python3 -m venv .venv `

`source .venv/bin/activate`

Instale os requirements

`pip install -r requirements.txt `

## Data
Nossos dados são arquivos que contém as laranjas detectadas e suas respectivas posições.

gt = ground true - a verdade, o gabartio. São os dados fornecidos pela Embrapa 

trackers = os trackers que iremos testar

Comando para rodar:

` python scripts/run_mot_challenge.py --BENCHMARK moranget --SPLIT_TO_EVAL test --TRACKERS_TO_EVAL myTracker `

Única coisa que irá mudar cada vez que formos testar é o `myTracker`.

Vamos colocar o nome que demos para a pasta onde está o tracker.

Estrutura da pasta de data

```
data/
├── gt/
│   ├── moranget/
│   │   ├── moranget-test/
│   │   │   ├── V01/
│   │   │   │   ├── geom/
│   │   │   │   ├── gt/
│   │   │   │   ├── images/
│   │   │   │   ├── database.db
│   │   │   │   ├── oranges-3d.txt
│   │   │   │   └── seqinfo.ini
│   │   └── seqmaps/
│   │       ├── moranget-test.txt
└── trackers/
│   ├── moranget/
│   │   ├── moranget-test/
│   │   │   ├── botSort/
│   │   │   ├── any_tracker/
│   │   │   ├── myTracker/
│   │   │   │   ├──data/
│   │   │   │   │   └── v01.txt

```

V01 é o nome do vídeo. Podemos testar quantos vídeos a gente quiser, só adicionar uma pasta no gt, o nome do video no txt dentro de seqmaps e um txt para esse vídeo no trackers.

O txt do vídeo no trackers contém as detecções e tracking do novo tracker que testamos. Deve estar no formato padrão definido pelo pessoal da Embrapa, que é:

| Quadro | Laranja # | BBox TLC (x) | BBox TLC (y) | Largura | Altura | Confianca | Tipo | Visível |  
| ----- | -------- | ------------ | ------------ | ----- | ------ | ---------- | ---- | ------- |
|1      | 5        | 780.06       | 621.58       | 70.83 | 70.82  | 1          | 1    | 1.0     |
|1      | 6        | 183.78       | 282.15       | 79.27 | 79.26  | 1          | 1    | 0.0     |

Os dois últimos valores coloquei sempre 1, pq não sei direito como são úteis. 
No gt, as 3 últimas colunas são sempre 1.


### YOLO Track
Tem que instalar o pacote da ultralytics para funcionar. É bem pesado.

`pip install ultralytics`

Tem um arquivo yolo_track.py que permite gerar o txt com as detecções e o tracking de um determinado modelo de YOLO e de tracker.

Ele já gera o arquivo no local certo no data/trackers/moranget/moranget-test/