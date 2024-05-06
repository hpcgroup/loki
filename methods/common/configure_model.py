from importlib import import_module

def get_h2o_args(parser):
    parser.add_argument("--use-h2o", action='store_true', default=False, help="use the H2O algos")
    parser.add_argument("--heavy-ratio", type=float, default=0.1, help="H2O heavy ratio," "set to 0.1 by default")
    return parser

def get_topk_args(parser):
    parser.add_argument("--use-topk", action='store_true', default=False, help="use the H2O algos")
    parser.add_argument("--top-k", type=float, default=-1, help="top k tokens to consider - >1 exact number, <1 fraction of tokens to consider," "set to -1 to use all tokens")
    return parser

def get_spar_args(parser):
    parser.add_argument("--use-sparq", action='store_true', default=False, help="use the Spar algos")
    parser.add_argument("--use-spark", action='store_true', default=False, help="use the Spar algos")
    parser.add_argument("--use-spar-hat", action='store_true', default=False, help="use the Spar Hat algo")
    parser.add_argument("--top-r", type=float, default=-1, help="top r channels to consider," "set to -1 to use all channels")
    return parser

def get_pca_args(parser):
    parser.add_argument("--use-pca", action='store_true', default=False, help="use the PCA algos")
    parser.add_argument("--use-pca-topk", action='store_true', default=False, help="use the PCA TopK algos")
    parser.add_argument("--rotary-type", type=str, default="postrotary", help="rotary type")
    parser.add_argument("--recent-ratio", type=float, default=-1, help="PcaTopK recent ratio," "set to -1 by default")
    return parser

def get_save_tensor_args(parser):
    parser.add_argument("--save-tensors", action='store_true', default=False, help="save tensor to file")
    parser.add_argument("--tensors-dir", type=str, default="./", help="file to save tensor to")
    return parser


def get_modifier(args):
    if args.use_topk:
        method_name = "top_k"
        module_name = ".baselines.topk.modify_" + args.model_type
    elif args.use_h2o:
        method_name = "h2o"
        module_name = ".baselines.h2o.modify_" + args.model_type
    elif args.use_sparq:
        method_name = "sparq"
        module_name = ".baselines.sparq.modify_" + args.model_type
    elif args.use_spark:
        method_name = "spark"
        module_name = ".spark.modify_" + args.model_type
    elif args.use_spar_hat:
        method_name = "sparhat"
        module_name = ".parhat.modify_" + args.model_type
    elif args.use_pca:
        method_name = "pca"
        module_name = ".pca.modify_" + args.model_type
    elif args.use_pca_topk:
        method_name = "pca_topk"
        module_name = ".pca_topk.modify_" + args.model_type
    else:
        return None
      
    method_name = "make_" + args.model_type + "_attention_" + method_name
    module = import_module(module_name, package="methods")
    method = getattr(module, method_name)
    return method

def get_config_dict(args):
    config_dict = {}
    if args.use_h2o:
        config_dict["method"] = "h2o"
        config_dict["heavy_ratio"] = args.heavy_ratio
    elif args.use_topk:
        config_dict["method"] = "topk"
        config_dict["top_k"] = args.top_k
    elif args.use_spark:
        config_dict["method"] = "spark"
        config_dict["top_r"] = args.top_r
        config_dict["top_k"] = args.top_k
    elif args.use_sparq:
        config_dict["method"] = "sparq"
        config_dict["top_r"] = args.top_r
        config_dict["top_k"] = args.top_k
    elif args.use_spar_hat:
        config_dict["method"] = "spar_hat"
        config_dict["top_r"] = args.top_r
    elif args.use_pca:
        config_dict["method"] = "pca"
        config_dict["top_r"] = args.top_r
    elif args.use_pca_topk:
        config_dict["method"] = "pca_topk"
        config_dict["top_r"] = args.top_r
        config_dict["top_k"] = args.top_k
        config_dict["rotary_type"] = args.rotary_type
        config_dict["recent_ratio"] = args.recent_ratio
    return config_dict