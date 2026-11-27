//! Transform Tensor terminal handlers.

use tc_error::TCError;

use super::super::{Tensor, schema::dtype};
use super::{axes, no_parameters, optional, parameter, shape, state};
use crate::{CollectionState, CopyFrom, StorageContext};

pub(super) struct Matmul<'a, Txn: StorageContext>(pub(super) &'a Tensor<Txn>);

impl<'a, S: CollectionState<Txn = Txn>, Txn: StorageContext> tc_ir::Handler<'a, S>
    for Matmul<'a, Txn>
{
    fn post<'txn>(self: Box<Self>) -> Option<tc_ir::PostHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, mut params| {
            Box::pin(async move {
                let right = parameter(&mut params, "r")?.into_tensor()?;
                no_parameters(&params)?;
                Ok(state(self.0.matmul(txn.id(), &right).await?))
            })
        }))
    }
}

unary_handler! {
    Mt => mt,
    Diag => diag,
    #[cfg(feature = "complex")]
    Mh => mh,
    #[cfg(feature = "complex")]
    Fft => fft,
    #[cfg(feature = "complex")]
    Ifft => ifft,
    #[cfg(feature = "complex")]
    Fft2 => fft2,
    #[cfg(feature = "complex")]
    Ifft2 => ifft2,
}

pub(super) struct Reshape<'a, Txn: StorageContext>(pub(super) &'a Tensor<Txn>);

impl<'a, S: CollectionState<Txn = Txn>, Txn: StorageContext> tc_ir::Handler<'a, S>
    for Reshape<'a, Txn>
{
    fn get<'txn>(self: Box<Self>) -> Option<tc_ir::GetHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, key| {
            Box::pin(async move {
                let _ = txn;
                Ok(state(self.0.clone().reshape(shape(S::from(key))?)?))
            })
        }))
    }
}

pub(super) struct Broadcast<'a, Txn: StorageContext>(pub(super) &'a Tensor<Txn>);

impl<'a, S: CollectionState<Txn = Txn>, Txn: StorageContext> tc_ir::Handler<'a, S>
    for Broadcast<'a, Txn>
{
    fn get<'txn>(self: Box<Self>) -> Option<tc_ir::GetHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, key| {
            Box::pin(async move {
                let _ = txn;
                Ok(state(self.0.clone().broadcast(shape(S::from(key))?)?))
            })
        }))
    }
}

pub(super) struct ExpandDims<'a, Txn: StorageContext>(pub(super) &'a Tensor<Txn>);

impl<'a, S: CollectionState<Txn = Txn>, Txn: StorageContext> tc_ir::Handler<'a, S>
    for ExpandDims<'a, Txn>
{
    fn get<'txn>(self: Box<Self>) -> Option<tc_ir::GetHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, key| {
            Box::pin(async move {
                let _ = txn;
                let key = S::from(key);
                let axes = if key.is_none() {
                    vec![self.0.shape().len()]
                } else {
                    axes(key)?
                };

                Ok(state(self.0.clone().unsqueeze(axes.into_iter().collect())?))
            })
        }))
    }
}

pub(super) struct Squeeze<'a, Txn: StorageContext>(pub(super) &'a Tensor<Txn>);

impl<'a, S: CollectionState<Txn = Txn>, Txn: StorageContext> tc_ir::Handler<'a, S>
    for Squeeze<'a, Txn>
{
    fn get<'txn>(self: Box<Self>) -> Option<tc_ir::GetHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, key| {
            Box::pin(async move {
                let _ = txn;
                Ok(state(
                    self.0
                        .clone()
                        .squeeze(axes(S::from(key))?.into_iter().collect())?,
                ))
            })
        }))
    }
}

pub(super) struct Flip<'a, Txn: StorageContext>(pub(super) &'a Tensor<Txn>);

impl<'a, S: CollectionState<Txn = Txn>, Txn: StorageContext> tc_ir::Handler<'a, S>
    for Flip<'a, Txn>
{
    fn get<'txn>(self: Box<Self>) -> Option<tc_ir::GetHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, key| {
            Box::pin(async move {
                let _ = txn;
                let axes = axes(S::from(key))?;
                let [axis] = axes.as_slice() else {
                    return Err(TCError::bad_request("flip requires exactly one axis"));
                };

                Ok(state(self.0.clone().flip(*axis)?))
            })
        }))
    }
}

pub(super) struct Cast<'a, Txn: StorageContext>(pub(super) &'a Tensor<Txn>);

impl<'a, S: CollectionState<Txn = Txn>, Txn: StorageContext> tc_ir::Handler<'a, S>
    for Cast<'a, Txn>
{
    fn get<'txn>(self: Box<Self>) -> Option<tc_ir::GetHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, key| {
            Box::pin(async move {
                let dtype = dtype(S::from(key).into_value()?)?;
                Ok(state(self.0.cast(txn.id(), dtype).await?))
            })
        }))
    }
}

pub(super) struct Transpose<'a, Txn: StorageContext>(pub(super) &'a Tensor<Txn>);

impl<'a, S: CollectionState<Txn = Txn>, Txn: StorageContext> tc_ir::Handler<'a, S>
    for Transpose<'a, Txn>
{
    fn get<'txn>(self: Box<Self>) -> Option<tc_ir::GetHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, key| {
            Box::pin(async move {
                let _ = txn;
                let key = S::from(key);
                Ok(state(self.0.clone().transpose(if key.is_none() {
                    None
                } else {
                    Some(axes(key)?.into_iter().collect())
                })?))
            })
        }))
    }

    fn post<'txn>(self: Box<Self>) -> Option<tc_ir::PostHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |_txn, mut params| {
            Box::pin(async move {
                let permutation = optional(&mut params, "perm")
                    .map(axes)
                    .transpose()?
                    .map(|axes| axes.into_iter().collect());

                no_parameters(&params)?;
                Ok(state(self.0.clone().transpose(permutation)?))
            })
        }))
    }
}

pub(super) struct AsDense<'a, Txn: StorageContext>(pub(super) &'a Tensor<Txn>);

impl<'a, S: CollectionState<Txn = Txn>, Txn: StorageContext> tc_ir::Handler<'a, S>
    for AsDense<'a, Txn>
{
    fn get<'txn>(self: Box<Self>) -> Option<tc_ir::GetHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, key| {
            Box::pin(async move {
                if !S::from(key).is_none() {
                    return Err(TCError::bad_request("as_dense expects no key"));
                }

                Ok(state(self.0.as_dense(txn.id()).await?))
            })
        }))
    }
}

pub(super) struct Copy<'a, Txn: StorageContext>(pub(super) &'a Tensor<Txn>);

impl<'a, S: CollectionState<Txn = Txn>, Txn: StorageContext> tc_ir::Handler<'a, S>
    for Copy<'a, Txn>
{
    fn get<'txn>(self: Box<Self>) -> Option<tc_ir::GetHandler<'a, 'txn, S>>
    where
        'txn: 'a,
    {
        Some(Box::new(move |txn, key| {
            Box::pin(async move {
                let _ = key;
                Ok(state(
                    crate::tensor::Tensor::copy_from(
                        txn,
                        txn.subcontext_unique().context().await?,
                        self.0,
                    )
                    .await?,
                ))
            })
        }))
    }
}
