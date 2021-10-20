from absl import app
from absl import flags
from utils.CrossValM2 import CrossValidationM2


FLAGS = flags.FLAGS

flags.DEFINE_string('filename', 'iris', 'Path to data file.')
flags.DEFINE_boolean('debug', False, 'Produces debugging output.')
flags.DEFINE_boolean('prune', False, 'True if pruning.')
flags.DEFINE_float('minsup', 0.01, 'Minimum support level')
flags.DEFINE_float('minconf', 0.5, 'Minimum confidence level')
flags.DEFINE_boolean('multiple', False, "Multiple minsup")


def main(argv):
    if FLAGS.debug:
        print("Non-flag arguments:", argv)
    data_path = f"./datasets/{FLAGS.filename}.data"
    scheme_path = f"./datasets/{FLAGS.filename}.names"
    print("data_path:", data_path)
    print("scheme_path:", scheme_path)
    minsup = FLAGS.minsup
    minconf = FLAGS.minconf

    validation = CrossValidationM2(data_path, scheme_path, minsup, minconf)

    print(f"Prune: {FLAGS.prune}, Multiple Minsup: {FLAGS.multiple}")
    validation.cross_validation(multiple=FLAGS.multiple,
                                prune=FLAGS.prune) # multiple minsups

if __name__ =="__main__":
    app.run(main)
