"""
Storage data of expression profile or GO
"""


class GeneDataCase:
    def __init__(self,
                 gene_id: str,
                 gene_alias: list = None,  # TODO: add this to <EPParsing>
                 gene_go: dict = None,
                 gene_expression_profile: dict = None):
        self.gene_id = gene_id
        self.gene_alias = gene_alias
        self.gene_go = gene_go
        self.gene_expression_profile = gene_expression_profile

    def __getitem__(self, item):
        return self.__dict__.get(item)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    @staticmethod
    def generate_key_name(values: list):
        return [f'organ{i}' for i in range(1, len(values)+1)]


if __name__ == '__main__':
    # === test data ====
    test_gene = GeneDataCase('a', gene_go={'go_id': 'b', 'go_name': 'c'},
                             gene_expression_profile={'liver': 0.3, 'kidney': 6})
    print(test_gene.__dict__)
    print(test_gene.gene_expression_profile)
    print(test_gene['gene_expression_profile'])
