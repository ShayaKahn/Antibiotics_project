class OtuFit:
    def __init__(self, ref_OTU, OTU, threshold):
        self.ref_OTU = ref_OTU
        self.OTU = OTU
        self.threshold = threshold

    def _find_intersection(self):
        species_list_ref_otu = list(self.ref_OTU['Species'])
        species_list_otu = list(self.OTU['Species'])
        common_species = list(set(species_list_ref_otu) & set(species_list_otu))
        return common_species

    def _find_fraction(self, common_species):
        row_sums_ref_OTU = self.ref_OTU.iloc[:, 1:].sum(axis=1)
        numerator_ref_OTU = row_sums_ref_OTU[self.ref_OTU.iloc[:, 0].isin(common_species)].sum()
        denominator_ref_OTU = row_sums_ref_OTU.sum()
        fraction_ref_OTU = numerator_ref_OTU / denominator_ref_OTU

        row_sums_OTU = self.OTU.iloc[:, 1:].sum(axis=1)
        numerator_OTU = row_sums_OTU[self.OTU.iloc[:, 0].isin(common_species)].sum()
        denominator_OTU = row_sums_OTU.sum()
        fraction_OTU = numerator_OTU / denominator_OTU
        return fraction_ref_OTU, fraction_OTU

    def _build_sub_OTU(self, common_species):
        sub_ref_OTU = self.ref_OTU[self.ref_OTU.iloc[:, 0].isin(common_species)]
        sub_OTU = self.OTU[self.OTU.iloc[:, 0].isin(common_species)]
        return sub_ref_OTU, sub_OTU

    def pipe(self):
        common_species = self._find_intersection()
        self.fraction_ref_OTU, self.fraction_OTU = self._find_fraction(common_species)
        try:
            assert(self.fraction_ref_OTU >= self.threshold)
        except AssertionError:
            print('Fraction of ref_OTU is less than the threshold')
        try:
            assert(self.fraction_OTU >= self.threshold)
        except AssertionError:
            print('Fraction of ref_OTU is less than the threshold')
        sub_ref_OTU, sub_OTU = self._build_sub_OTU(common_species)
        return sub_ref_OTU, sub_OTU