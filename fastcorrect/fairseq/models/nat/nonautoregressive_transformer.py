# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATDecoder, FairseqNATModel, ensemble_decoder
from fairseq.models.transformer import Embedding, Embeddingright
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.modules import FairseqDropout
from torch import Tensor

def _mean_pooling(enc_feats, src_masks):
    # enc_feats: T x B x C
    # src_masks: B x T or None
    if src_masks is None:
        enc_feats = enc_feats.mean(0)
    else:
        src_masks = (~src_masks).transpose(0, 1).type_as(enc_feats)
        enc_feats = (
            (enc_feats / src_masks.sum(0)[None, :, None]) * src_masks[:, :, None]
        ).sum(0)
    return enc_feats


def _argmax(x, dim):
    return (x == x.max(dim, keepdim=True)[0]).type_as(x)


@register_model("nonautoregressive_transformer")
class NATransformerModel(FairseqNATModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        if getattr(args, 'remove_edit_emb', False):
            print("Remove edit emb!")
            self.remove_edit_emb = True
        else:
            self.remove_edit_emb = False

        self.to_be_edited_size = getattr(args, "to_be_edited_size", 1)

        if getattr(args, 'assist_edit_loss', False):
            print("add assist edit loss!")
            self.assist_edit_loss = True
        else:
            self.assist_edit_loss = False

        self.werdur_max_predict = getattr(args, 'werdur_max_predict', 5.0)

        self.werdur_loss_type = getattr(args, 'werdur_loss_type', 'l2')

        if self.werdur_loss_type == 'l2':
            self.werdur_loss_func = F.mse_loss
        else:
            raise ValueError("Unsupported werdur_loss_type")


    @property
    def allow_length_beam(self):
        return True

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)

        parser.add_argument(
            "--remove-edit-emb",
            action="store_true",
            default=False,
            help="whether to remove edit emb",
        )
        parser.add_argument(
            "--assist-edit-loss",
            action="store_true",
            default=False,
            help="whether to use assist edit loss",
        )
        parser.add_argument(
            "--pred-length-offset",
            action="store_true",
            help="predicting the length difference between the target and source sentences",
        )
        parser.add_argument(
            "--length-loss-factor",
            type=float,
            help="weights on the length prediction loss",
        )
        parser.add_argument(
            "--edit-emb-dim",
            type=int,
            help="dimension of edit emb",
        )
        parser.add_argument(
            "--to-be-edited-size",
            type=int,
            help="size of to be edited",
        )
        parser.add_argument(
            "--werdur-max-predict",
            type=float,
            help="dimension of edit emb",
        )
        parser.add_argument(
            "--werdur-loss-type",
            type=str,
            help="type of werdur loss",
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = NATransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history
        to_be_edited_pred = decoder_out.to_be_edited_pred
        wer_dur_pred = decoder_out.wer_dur_pred

        for_wer_gather = wer_dur_pred.cumsum(dim=-1)
        for_wer_gather = torch.nn.functional.one_hot(for_wer_gather, num_classes=for_wer_gather.max() + 1)[:, :-1, :-1].sum(-2).cumsum(dim=-1)

        # execute the decoder
        output_masks = output_tokens.ne(self.pad)
        _scores, _tokens = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=step,
            wer_dur=wer_dur_pred,
            to_be_edited=to_be_edited_pred, for_wer_gather=for_wer_gather
        ).max(-1)

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])
        if history is not None:
            history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )

    def initialize_output_tokens(self, encoder_out, src_tokens):
        if True:
            if True:
                if True:
                    wer_dur_pred, _, _= self.decoder.forward_wer_dur_and_tbe(
                        normalize=False, encoder_out=encoder_out
                    )

                    wer_dur_pred = wer_dur_pred.squeeze(-1).round().long().clamp_(min=0)
                    length_tgt = wer_dur_pred.sum(-1)


        max_length = length_tgt.clamp_(min=2).max()

        idx_length = utils.new_arange(src_tokens, max_length)

        initial_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out.encoder_out)

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
            to_be_edited_pred=None,
            wer_dur_pred=wer_dur_pred
        ), encoder_out

    def regenerate_length_beam(self, decoder_out, beam_size):
        output_tokens = decoder_out.output_tokens
        length_tgt = output_tokens.ne(self.pad).sum(1)
        length_tgt = (
            length_tgt[:, None]
            + utils.new_arange(length_tgt, 1, beam_size)
            - beam_size // 2
        )
        length_tgt = length_tgt.view(-1).clamp_(min=2)
        max_length = length_tgt.max()
        idx_length = utils.new_arange(length_tgt, max_length)

        initial_output_tokens = output_tokens.new_zeros(
            length_tgt.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(decoder_out.output_scores)

        return decoder_out._replace(
            output_tokens=initial_output_tokens, output_scores=initial_output_scores
        )

######################
# fastspeech modules
######################
class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.
    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1, eps=1e-12):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=eps)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)


class DurationPredictor(torch.nn.Module):
    def __init__(self, idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, ffn_layers=1, offset=1.0, ln_eps=1e-12, remove_edit_emb=False, to_be_edited_size=1, add_glo_biclass=False, padding='SAME'):
        """Initilize duration predictor module.
        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
            offset (float, optional): Offset value to avoid nan in log domain.
        """
        super(DurationPredictor, self).__init__()
        #'''
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        self.remove_edit_emb = remove_edit_emb
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=0),
                torch.nn.ReLU(),
                LayerNorm(n_chans, dim=1, eps=ln_eps),
                FairseqDropout(dropout_rate, module_name="DP_dropout")
            )]
        if ffn_layers == 1:
            self.werdur_linear = torch.nn.Linear(n_chans, 1)
            if self.add_glo_biclass:
                self.glo_biclass_linear = torch.nn.Linear(n_chans, 1)
            if not self.remove_edit_emb:
                self.edit_linear = torch.nn.Linear(n_chans, to_be_edited_size)
        else:
            assert ffn_layers == 2
            self.werdur_linear = torch.nn.Sequential(
                torch.nn.Linear(n_chans, n_chans // 2),
                torch.nn.ReLU(),
                FairseqDropout(dropout_rate, module_name="DP_dropout"),
                torch.nn.Linear(n_chans // 2, 1),
            )
            if not self.remove_edit_emb:
                self.edit_linear = torch.nn.Sequential(
                    torch.nn.Linear(n_chans, n_chans // 2),
                    torch.nn.ReLU(),
                    FairseqDropout(dropout_rate, module_name="DP_dropout"),
                    torch.nn.Linear(n_chans // 2, to_be_edited_size),
                )


    def forward(self, xs, x_nonpadding=None):
        #'''
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            if self.padding == 'SAME':
                xs = F.pad(xs, [self.kernel_size // 2, self.kernel_size // 2])
            elif self.padding == 'LEFT':
                xs = F.pad(xs, [self.kernel_size - 1, 0])
            xs = f(xs)  # (B, C, Tmax)
            if x_nonpadding is not None:
                xs = xs * x_nonpadding[:, None, :]

        xs = xs.transpose(1, -1)
        #'''
        werdur = self.werdur_linear(xs) * x_nonpadding[:, :, None]  # (B, Tmax)
        if not self.remove_edit_emb:
            to_be_edited = self.edit_linear(xs) * x_nonpadding[:, :, None]  # (B, Tmax)
        else:
            to_be_edited = None

        return werdur, to_be_edited


class NATransformerDecoder(FairseqNATDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()

        self.encoder_embed_dim = args.encoder_embed_dim
        self.sg_length_pred = getattr(args, "sg_length_pred", False)
        self.pred_length_offset = getattr(args, "pred_length_offset", False)
        self.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
        self.to_be_edited_size = getattr(args, "to_be_edited_size", 1)
        if getattr(args, 'remove_edit_emb', False):
            # print("Remove edit emb!")
            self.remove_edit_emb = True
        else:
            self.remove_edit_emb = False
        if getattr(args, 'assist_edit_loss', False):
            # print("Remove edit emb!")
            self.assist_edit_loss = True
        else:
            self.assist_edit_loss = False
        self.edit_emb_dim = getattr(args, "edit_emb_dim", self.encoder_embed_dim // 2)


        if True:
            self.dur_predictor = DurationPredictor(idim=self.encoder_embed_dim, n_layers=5, n_chans=self.encoder_embed_dim, ffn_layers=2, ln_eps=1e-5, remove_edit_emb=(False or (self.remove_edit_emb and not self.assist_edit_loss)), to_be_edited_size=self.to_be_edited_size)
            if not self.remove_edit_emb:
                assert self.to_be_edited_size == 1, "to_be_edited_size not 1 when not remove_edit_emb is not implement"
                self.edit_embedding = Embeddingright(2, self.edit_emb_dim, None) if self.edit_emb_dim != 1 else torch.nn.Identity()
                self.reshape_weight = Embeddingright(self.encoder_embed_dim, self.encoder_embed_dim + self.edit_emb_dim, None)
                #self.reshape_weight = Embedding(self.encoder_embed_dim, self.encoder_embed_dim + 1, None)
            assert not getattr(args, "use_wer_dur", False)


    @ensemble_decoder
    def forward(self, normalize, encoder_out, prev_output_tokens, step=0, wer_dur=None, to_be_edited=None, for_wer_gather=None, debug_src_tokens=None, debug_tgt_tokens=None, **unused):

        features, _ = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            wer_dur=wer_dur,
            to_be_edited=to_be_edited, for_wer_gather=for_wer_gather
        )
        decoder_out = self.output_layer(features)
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out

    @ensemble_decoder
    def forward_length(self, normalize, encoder_out):
        enc_feats = encoder_out.encoder_out  # T x B x C
        src_masks = encoder_out.encoder_padding_mask  # B x T or None
        enc_feats = _mean_pooling(enc_feats, src_masks)
        if self.sg_length_pred:
            enc_feats = enc_feats.detach()
        length_out = F.linear(enc_feats, self.embed_length.weight)
        return F.log_softmax(length_out, -1) if normalize else length_out


    @ensemble_decoder
    def forward_wer_dur_and_tbe(self, normalize, encoder_out):
        enc_feats = encoder_out.encoder_out  # T x B x C
        src_masks = encoder_out.encoder_padding_mask  # B x T or None
        encoder_embedding = encoder_out.encoder_embedding  # B x T x C or B, T, nbest, C
        enc_feats = enc_feats.transpose(0, 1)
        # enc_feats = _mean_pooling(enc_feats, src_masks)

        src_masks = (~src_masks)
        if True:
            if True:
                wer_dur_out, to_be_edited_out = self.dur_predictor(enc_feats, src_masks)
        return wer_dur_out, to_be_edited_out, None

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        early_exit=None,
        wer_dur=None,
        to_be_edited=None, for_wer_gather=None,
        **unused
    ):
        """
        Similar to *forward* but only return features.

        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        # embedding
        if True:
            src_embd = encoder_out.encoder_embedding
            src_mask = encoder_out.encoder_padding_mask
            src_mask = (
                ~src_mask
                if src_mask is not None
                else prev_output_tokens.new_ones(*src_embd.size()[:2]).bool()
            )
            if True:
                if True:
                    if True:
                        x, decoder_padding_mask = self.forward_embedding(
                            prev_output_tokens,
                            self.forward_wer_dur_embedding(
                                src_embd, src_mask, prev_output_tokens.ne(self.padding_idx), wer_dur, to_be_edited, for_wer_gather
                            ),
                        )

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        for i, layer in enumerate(self.layers):

            # early exit from the decoder.
            if (early_exit is not None) and (i >= early_exit):
                break

            x, attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states}

    def forward_embedding(self, prev_output_tokens, states=None):
        # embed positions
        positions = (
            self.embed_positions(prev_output_tokens)
            if self.embed_positions is not None
            else None
        )

        # embed tokens and positions
        if states is None:
            x = self.embed_scale * self.embed_tokens(prev_output_tokens)
            if self.project_in_dim is not None:
                x = self.project_in_dim(x)
        else:
            x = states

        if positions is not None:
            x += positions
        x = self.dropout_module(x)
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        return x, decoder_padding_mask


    def forward_wer_dur_embedding(self, src_embeds, src_masks, tgt_masks, wer_dur, to_be_edited, for_wer_gather=None, debug_src_tokens=None, debug_tgt_tokens=None):
        # src_embeds: [B, T, C] * [s_T, t_T]

        for_wer_gather = for_wer_gather[:, :, None].long()

        to_reshape = torch.gather(src_embeds, 1, for_wer_gather.repeat(1, 1, src_embeds.shape[2]))

        to_reshape = to_reshape * tgt_masks[:, :, None]
        return to_reshape


    def forward_length_prediction(self, length_out, encoder_out, tgt_tokens=None):
        enc_feats = encoder_out.encoder_out  # T x B x C
        src_masks = encoder_out.encoder_padding_mask  # B x T or None
        if self.pred_length_offset:
            if src_masks is None:
                src_lengs = enc_feats.new_ones(enc_feats.size(1)).fill_(
                    enc_feats.size(0)
                )
            else:
                src_lengs = (~src_masks).transpose(0, 1).type_as(enc_feats).sum(0)
            src_lengs = src_lengs.long()

        if tgt_tokens is not None:
            # obtain the length target
            tgt_lengs = tgt_tokens.ne(self.padding_idx).sum(1).long()
            if self.pred_length_offset:
                length_tgt = tgt_lengs - src_lengs + 128
            else:
                length_tgt = tgt_lengs
            length_tgt = length_tgt.clamp(min=0, max=255)

        else:
            # predict the length target (greedy for now)
            # TODO: implementing length-beam
            pred_lengs = length_out.max(-1)[1]
            if self.pred_length_offset:
                length_tgt = pred_lengs - 128 + src_lengs
            else:
                length_tgt = pred_lengs

        return length_tgt


@register_model_architecture(
    "nonautoregressive_transformer", "nonautoregressive_transformer"
)
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)


@register_model_architecture(
    "nonautoregressive_transformer", "nonautoregressive_transformer_wmt_en_de"
)
def nonautoregressive_transformer_wmt_en_de(args):
    base_architecture(args)
