## How to run
```bash
$ conda create -n <env_name> python=3.10
$ conda activate <env_name>
```

```bash
$ pip install -r requirements.txt
```

```bash
$ python main.py
```

## How to use 
- Press Space to toggle play/pause the solving process.
- Press R to reset the map to initial state.

## Run the algorithms separately
```
python search.py --input <input_file> --transition <transition_type> --type <type> --timeout <timeout>
```

For example, to run the A* algorithm with 'box-moving' transition on the 0_demo_00.txt file with a timeout of 120 seconds, run the following command:

```
python search.py --input official_test/0_demo_00.txt --transition box --type 
A* --timeout 120
```

