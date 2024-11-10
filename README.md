## Demo video 
[![Demo video](https://img.youtube.com/vi/HoPs7_Rpnqg/maxresdefault.jpg)](https://www.youtube.com/watch?v=HoPs7_Rpnqg)

## How to run
- Clone the repository
```bash
$ git clone https://github.com/ThisIsNotDat/Sokoban.git
$ cd Sokoban
```

- Create a virtual environment and install the dependencies
```bash
$ conda create -n <env_name> python=3.12
$ conda activate <env_name>
```

```bash
$ pip install -r requirements.txt
```

### Run the game
- Run the `main.py` file
```bash
$ python main.py
```

### Run the search algorithms
- To run the search algorithms directly, run the following command.
```bash
python search.py --input <input_file> --transition <transition_type> --type <
type> --timeout <timeout>
```
- For example, to run the A* algorithm with 'box-moving' transition model on the test `demo_test/0_demo_00.txt`
with the time limit of 120 seconds, you can run
```bash
python search.py --input official_test/0_demo_00.txt --transition box --type A*
--timeout 120
```

- The result of running `search.py` would be JSON files in the following format:
```json
{
    "node": "llDD",
    "node_number": 2,
    "cost": 8,
    "time": 0.017000198364257812,
    "current_memory": 434629,
    "peak_memory": 442627
}
```

- To generate the `.txt` output file like the requirement, please run search_all.py. This will execute
all types of transition models and algorithms for a single input file and combine the result into
2 text files, each being the corresponding output of a transition model. For example, to run
algorithms on the test `demo_test/0_demo_00.txt`, you can run
```bash
python search_all.py --input demo_test/0_demo_00.txt
```
The result would be stored in two text files:
- `demo_test/output/0_demo_00 _ares_output.txt`
- `demo_test/output/0_demo_00_box_output.txt`
, which is output of 'Ares-moving' and 'box-moving', respectively.

## How to use 
Use the buttons on GUI, or the following keys to interact with the game:
- Press Space to toggle play/pause the solving process.
- Press R to reset the map to initial state.
- Press ESC to exit the game.

## Acknowledgements
Thanks to the following authors for their assets:
- [dani_maccari](https://dani-maccari.itch.io/sokoban-tileset) for the Sokoban tileset.
- [jinxit4](https://civitai.com/models/375001/rpg-top-down-4-direction-walk-cycle-pony) for the Peach Princess character sprite.