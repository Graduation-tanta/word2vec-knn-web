# word2vec-implementation
Implementation of word2vec as in Tensorflow examples on a piece of wikipedia dataset dumps, the model generated a 500MB array as a pickle file That will be uploaded on drive later on.
It has some good results for finding the K nearest words to a given word. 
aslo associated with python web server which will respond to queries and responds with an animated canvas that describes the words similarities.

I used these tutorials:
----------------------------------------------------------------------------------------
http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/

http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/5_word2vec.ipynb    

http://distill.pub/2016/misread-tsne/
----------------------------------------------------------------------------------------
Some examples of k nearest words:
----------------------------------------------------------------------------------------

Nearest to machine: ['gun', 'light', 'artillery', 'battery', 'rifle', 'tank', 'engineers', 'heavy']

Nearest to dynamic: ['problem', 'models', 'method', 'applications', 'systems', 'allows', 'theory', 'users']

Nearest to surrounding: ['adjacent', 'nearby', 'site', 'walls', 'buildings', 'areas', 'timber', 'importance']

Nearest to four: ['five', 'seven', 'three', 'second', 'six', 'eight', 'half', 'two']

Nearest to cat: ['dog', 'dragon', 'monster', 'adventure', 'ds', 'bit', 'playstation', 'fun']

Nearest to egg: ['fish', 'eggs', 'nest', 'rice', 'typically', 'species', 'usually', 'often']

Nearest to batman: ['comics', 'evil', 'kill', 'characters', 'heroes', 'character', 'comic', 'escape']

Nearest to blood: ['cells', 'disease', 'brain', 'patients', 'treatment', 'drug', 'humans', 'cell']

Nearest to stream: ['creek', 'watershed', 'mouth', 'flows', 'river', 'reaches', 'tributary', 'township']

Nearest to hitler: ['nazi', 'jews', 'wilhelm', 'german', 'von', 'prisoners', 'germany', 'allied']

Nearest to april: ['february', 'december', 'september', 'october', 'november', 'august', 'june', 'may']

Nearest to traffic: ['junction', 'roads', 'temporary', 'si', 'regulations', 'trunk', 'prohibition', 'amendment']

Nearest to khalil: ['ali', 'fêted', 'sablé', 'caggiano', 'ye', 'sunnybank', 'gadzhikhanov', 'ahmed']

Nearest to king: ['prince', 'bc', 'reign', 'princess', 'crown', 'empire', 'duke', 'emperor']

Nearest to cancer: ['disease', 'patients', 'cells', 'brain', 'gene', 'treatment', 'surgery', 'clinical']

Nearest to league: ['premier', 'cup', 'football', 'champions', 'clubs', 'uefa', 'club', 'goals']

Nearest to book: ['press', 'books', 'philosophy', 'publication', 'stories', 'fiction', 'journal', 'translated']

Nearest to nvidia: ['devices', 'incompetency', 'azin', 'crystals', 'lowitz', 'padiglione', 'kilmallie', 'benefactive']

Nearest to obama: ['president', 'bush', 'presidential', 'clinton', 'visit', 'prime', 'trump', 'barack']

Nearest to story: ['stories', 'novel', 'characters', 'comic', 'friends', 'cast', 'character', 'secret']

Nearest to mind: ['things', 'something', 'why', 'let', 'you', 'want', 'true', 'know']

Nearest to marvel: ['comics', 'character', 'comic', 'characters', 'evil', 'escape', 'disney', 'universe']

Nearest to universe: ['earth', 'miss', 'comics', 'characters', 'evil', 'beauty', 'pageant', 'marvel']

Nearest to uncle: ['daughter', 'brother', 'wife', 'son', 'father', 'died', 'friend', 'younger']

Nearest to man: ['woman', 'boy', 'my', 'fan', 'girl', 'dragon', 'secret', 'kill']

Nearest to fantastic: ['fantasy', 'thomachan', 'chl', 'theme', 'bad', 'boy', 'condiescu', 'tatoul']

Nearest to usually: ['typically', 'often', 'types', 'sometimes', 'likely', 'commonly', 'these', 'or']

Nearest to sunday: ['saturday', 'friday', 'pm', 'monday', 'thursday', 'wednesday', 'tuesday', 'am']

Nearest to god: ['holy', 'faith', 'jesus', 'christ', 'ancient', 'churches', 'priest', 'our']

Nearest to technology: ['institute', 'systems', 'computer', 'engineering', 'science', 'research', 'technologies', 'applications']
