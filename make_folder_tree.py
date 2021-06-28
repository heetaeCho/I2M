import os


bases = ["model/", 'result/', "model_tt/", 'result_tt/']
##bases = ['result/']
projects = ['npp/', 'komodo/', 'vscode/']
data_types = ['title/', 'body/']
models = ['cnn/' , 'rnn/']
embedding_types = ['e/', 'w/', 'g/','f/']

for base in bases:
    os.mkdir(base)
    
    for project in projects:
        os.mkdir(base+project)
        if base == 'model/' or base == 'model_tt/':
            continue
        
        for data_type in data_types:
            os.mkdir(base+project+data_type)

            for model in models:
                os.mkdir(base+project+data_type+model)

                for embedding in embedding_types:
                    os.mkdir(base+project+data_type+model+embedding)
