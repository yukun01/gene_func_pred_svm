"""
Parse gene expression profile by using this file
"""

from data_storage import GeneDataCase


class GOParsing:
    def __init__(self, handle, header: bool):
        self.handle = handle
        self.header = header

    def parse(self):
        if self.header:
            self.handle.__next__()

        for line in self.handle:
            line = line.rstrip().split('\t')
            yield line


class EPParsing:
    def __init__(self, handle=None, header: bool = None):
        self.handle = handle
        self.header = header
        self.anonymous = 1

    def parse(self):
        if self.header:
            organ_name = self.handle.__next__().rstrip().split('\t')[1:]
            yield organ_name

        for line in self.handle:
            line = line.rstrip().split('\t')
            if not line[0]:
                line[0] = f'anonymous_gene_{self.anonymous}'
                self.anonymous += 1
            yield line


class GeneIO:
    def __init__(self):
        self.gene_data = {}  # key = gene_id, value = gene_info <GeneDataCase>

    def __getitem__(self, item):
        return self.__dict__.get(item)

    def parse(self, handle, format, header=True):
        """
        calling this function by GeneIO.parse(handle, format)
        GO file format example: entrez_id (or other gene id)   go_id   go_name
                                ... ... ...
        EP file format example: entrez_id (or other gene id)    organ1  organ2 ...
                                ... ... ... ...
        :param handle: open("filepath")
        :param format: GO: GeneOntology or EP: ExpressionProfile
        :return: gene_data
        """
        if format == 'GO':
            parse = GOParsing(handle=handle, header=header).parse()
            for gene_id, go_id, go_name in parse:
                if gene_id not in self.gene_data:
                    self.gene_data[gene_id] = GeneDataCase(gene_id, gene_go={'go_id': [go_id], 'go_name': [go_name]})
                else:
                    self.gene_data[gene_id]['gene_go']['go_id'].append(go_id)
                    self.gene_data[gene_id]['gene_go']['go_name'].append(go_name)
        elif format == 'EP':
            parse = EPParsing(handle=handle, header=header).parse()
            if header:
                organ_name = parse.__next__()
            else:
                organ_name = ''
            for gene_id, *args in parse:
                if gene_id not in self.gene_data and not organ_name:
                    self.gene_data[gene_id] = GeneDataCase(gene_id, gene_expression_profile=dict(
                        zip(GeneDataCase.generate_key_name(args), args)))
                elif gene_id not in self.gene_data and organ_name:
                    self.gene_data[gene_id] = GeneDataCase(gene_id, gene_expression_profile=dict(
                        zip(organ_name, args)
                    ))
                elif gene_id in self.gene_data and not organ_name:
                    self.gene_data[gene_id]['gene_expression_profile'] = dict(
                        zip(GeneDataCase.generate_key_name(args), args))
                else:  # elif gene_id in self.gene_data and organ_name:
                    self.gene_data[gene_id]['gene_expression_profile'] = dict(
                        zip(organ_name, args)
                    )

    def _save(self, filepath):
        import numpy as np
        np.save(filepath, self.gene_data)


if __name__ == '__main__':
    go_handle = open('test_data/GO.txt')
    ep_handle = open('test_data/EP.txt')
    data_parse = GeneIO()
    data_parse.parse(handle=go_handle, format='GO')
    data_parse.parse(handle=ep_handle, format='EP')
    # print(data_parse.gene_data)
    print(data_parse.gene_data['80328'].__dict__)
    data_parse._save('./file/gene_data.npy')
