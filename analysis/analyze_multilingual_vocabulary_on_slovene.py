import pickle


with open('slovene_in_multilingual_vocabulary.pkl', 'rb') as f:
    dict = pickle.load(f)

    print('### DICT LEN ###')
    print(len(dict))
    el_under_10 = 0
    el_under_100 = 0
    el_under_1000 = 0
    el_under_10000 = 0
    el_under_100000 = 0
    el_under_1000000 = 0

    word_parts_count = 3181452926.0
    word_parts_num = 0
    for k, v in dict.items():
        word_parts_num += v
        if dict[k] < word_parts_count / 100000000:
            el_under_10 += 1
        if dict[k] < word_parts_count / 10000000:
            el_under_100 += 1
        if dict[k] < word_parts_count / 1000000:
            el_under_1000 += 1
        if dict[k] < word_parts_count / 100000:
            el_under_10000 += 1
        if dict[k] < word_parts_count / 10000:
            el_under_100000 += 1
        if dict[k] < word_parts_count / 1000:
            el_under_1000000 += 1
    print('### DICT OVER ' + str(100.0 / 100000000) + '% ###')
    print(len(dict) - el_under_10)
    print('### DICT OVER ' + str(100.0 / 10000000) + '%  ###')
    print(len(dict) - el_under_100)
    print('### DICT OVER ' + str(100.0 / 1000000) + '%  ###')
    print(len(dict) - el_under_1000)
    print('### DICT OVER ' + str(100.0 / 100000) + '%  ###')
    print(len(dict) - el_under_10000)
    print('### DICT OVER ' + str(100.0 / 10000) + '%  ###')
    print(len(dict) - el_under_100000)
    print('### DICT OVER ' + str(100.0 / 1000) + '%  ###')
    print(len(dict) - el_under_1000000)

    print('### WORD PARTS NUMBER ###')
    print(word_parts_num)

    el_under_10 = 0
    el_under_100 = 0
    el_under_1000 = 0
    el_under_10000 = 0
    el_under_100000 = 0
    el_under_1000000 = 0
    word_parts_num = 0
    for k, v in dict.items():
        word_parts_num += v
        if dict[k] < 10:
            el_under_10 += 1
        if dict[k] < 100:
            el_under_100 += 1
        if dict[k] < 1000:
            el_under_1000 += 1
        if dict[k] < 10000:
            el_under_10000 += 1
        if dict[k] < 100000:
            el_under_100000 += 1
        if dict[k] < 1000000:
            el_under_1000000 += 1
    print('### DICT OVER 10 LEN ###')
    print(len(dict) - el_under_10)
    print('### DICT OVER 100 LEN ###')
    print(len(dict) - el_under_100)
    print('### DICT OVER 1000 LEN ###')
    print(len(dict) - el_under_1000)
    print('### DICT OVER 10000 LEN ###')
    print(len(dict) - el_under_10000)
    print('### DICT OVER 100000 LEN ###')
    print(len(dict) - el_under_100000)
    print('### DICT OVER 1000000 LEN ###')
    print(len(dict) - el_under_1000000)
    #
    # print('### WORD PARTS NUMBER ###')
    # print(word_parts_num)

    print('test')
