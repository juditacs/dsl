import json
from sys import stdin


def main():
    groups = {
        'south_slavic': ['bs', 'hr', 'sr'],
        'indonesian': ['id', 'my'],
        'north_slavic': ['cz', 'sk'],
        'portuguese': ['pt-BR', 'pt-PT'],
        'spanish': ['es-AR', 'es-ES'],
        'xx': ['xx'],
        'cyrillic': ['bg', 'mk'],
    }
    for l in stdin:
        sumA = 0
        sumN = 0
        fs = l.strip().split('\t')
        stat = json.loads(fs[-1])
        for group, members in sorted(groups.items()):
            N = 0
            acc = 0
            for lang in members:
                if not lang in stat:
                    continue
                acc += stat[lang].get(lang, 0)
                N += sum(stat[lang].values())
                sumN += N
                sumA += acc
            if N > 0:
                print(', '.join(fs[:-1]))
                print('{0}: {1} {2}'.format(group, ','.join(members), float(acc) / N))
        print('Acc: {0}'.format(float(sumA) / sumN))
        print('==============================')

if __name__ == '__main__':
    main()
