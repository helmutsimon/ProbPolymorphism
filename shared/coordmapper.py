# coding=utf-8


def CoordMapper(lo):
    def process_row(chrom_pos):
        chrom, pos = chrom_pos
        new = lo.convert_coordinate(chrom, int(pos))
        if not new or len(new) > 1:
            return None

        new = new[0]
        if chrom != new[0]:
            # different chrom
            return None

        new_pos = new[1]

        return new_pos

    return process_row


