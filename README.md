# DSAIL Repository Template

![DSAIL_Character](https://user-images.githubusercontent.com/40286691/152486968-670af245-903e-4743-af70-d87531567721.PNG)

[DSAIL @ KAIST](https://dsail.kaist.ac.kr/)


```bash
.
├── configs ('--F', help='for configurations')
│   └── params.json ('--f', help='best parameters of dataset')
├── embedder.py ('--f', help='embedder file to control whole process')
├── main.py ('--f', help='main file of the process')
├── models ('--F', help='for models and baselines')
│   └── __init__.py
├── README.md ('--f', help='README for Github')
├── results ('--F', help='for the output of model and baselines')
├── sh ('--F', help='for shell files (e.g. experiments, parameter search)')
│   └── run.sh ('--f', help='shell files (e.g. experiments, parameter search)')
└── src ('--F', help='for utility functions')
    ├── __init__.py
    ├── argument.py ('--f', help='input arguments parser')
    ├── data.py ('--f', help='data loading related')
    └── utils.py ('--f', help='helper functions')
```

* F: Folder
* f: File

## How to use `.gitignore`

The `.gitignore` file is to specify intentionally untracked files to ignore. For example, the files are

* Too big (e.g. > 100mb)
* Confidential
* Not related to the project

Please refer to the 'PATTERN FORMAT' in git-scm document: [Guide](https://git-scm.com/docs/gitignore).

